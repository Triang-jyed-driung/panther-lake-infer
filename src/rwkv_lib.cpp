/*
 * RWKV-7 "Goose" CPU Inference Library
 * ======================================
 *
 * All model dimensions are read from the binary header at load time.
 * No hard-coded constants — works with any RWKV-7 model size.
 *
 * This file is compiled TWICE inside one translation unit by
 * src/rwkv_api.cpp — once in `namespace Fp16` (no RWKV_WEIGHTS_BF16)
 * and once in `namespace Bf16` (with it) — so a single build supports
 * both fp16 ("RWKV7") and bf16 ("RWKV7B") weight binaries.  The C API
 * below is compiled only when RWKV_ENGINE_EMBEDDED is NOT defined;
 * src/rwkv_api.cpp provides the top-level dispatch instead.
 *
 * Direct (single-format) build for backward compatibility:
 *   g++ -O3 -march=native -fopenmp -shared -fPIC -DRWKV_BUILD_DLL \
 *       -o librwkv.so rwkv_lib.cpp gemm_fp16.cpp -lm
 */

#define h_size 64
#include "rwkv.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>

#include <omp.h>
#include <immintrin.h>

#include "gemv_f16.h"
#include "gemm_fp16.h"
#include "utils_16.h"

/* 16-bit ↔ fp32 conversions live in utils_16.{h,cpp} (unified names). */

/* ── sigmoid ────────────────────────────────────────────────────── */
static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

/* ── AVX2 exp / sigmoid / tanh (vectorized element-wise math) ──────
 * cephes-style: exp(x) = 2^n · exp(g), g = f·ln2 with hi/lo split,
 * degree-7 Horner on g.  Arg clamped to ±87.5 so the 2^n exponent
 * trick stays in normal range.  Error ≈ 0.1 ulp.
 * ALL forward paths (decode / prefill / window) use these same
 * helpers so their outputs stay bit-consistent. */
static inline __m256 exp256_ps(__m256 x)
{
    const __m256 clamp_hi = _mm256_set1_ps( 87.5f);
    const __m256 clamp_lo = _mm256_set1_ps(-87.5f);
    x = _mm256_min_ps(_mm256_max_ps(x, clamp_lo), clamp_hi);

    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    __m256 y = _mm256_mul_ps(x, log2e);
    __m256 n = _mm256_round_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 f = _mm256_sub_ps(y, n);                  /* f ∈ [-0.5, 0.5] */

    /* g = f·ln2, hi/lo split for accurate reduction */
    const __m256 c1 = _mm256_set1_ps(0.693359375f);
    const __m256 c2 = _mm256_set1_ps(-2.12194440e-4f);
    __m256 g = _mm256_add_ps(_mm256_mul_ps(f, c1), _mm256_mul_ps(f, c2));

    /* exp(g) = 1 + g·h ,  h = Horner(1/5040 … 1/2, 1) */
    __m256 h = _mm256_set1_ps(1.0f / 5040.0f);
    h = _mm256_fmadd_ps(h, g, _mm256_set1_ps(1.0f / 720.0f));
    h = _mm256_fmadd_ps(h, g, _mm256_set1_ps(1.0f / 120.0f));
    h = _mm256_fmadd_ps(h, g, _mm256_set1_ps(1.0f / 24.0f));
    h = _mm256_fmadd_ps(h, g, _mm256_set1_ps(1.0f / 6.0f));
    h = _mm256_fmadd_ps(h, g, _mm256_set1_ps(0.5f));
    h = _mm256_fmadd_ps(h, g, _mm256_set1_ps(1.0f));
    __m256 eg = _mm256_fmadd_ps(g, h, _mm256_set1_ps(1.0f));

    /* scale by 2^n via exponent-bit arithmetic (n ∈ [-126, 126]) */
    __m256i ni = _mm256_cvtps_epi32(n);
    __m256 scale = _mm256_castsi256_ps(
        _mm256_add_epi32(_mm256_slli_epi32(ni, 23), _mm256_set1_epi32(0x3f800000)));
    return _mm256_mul_ps(eg, scale);
}

// ===========================================================================
// 对齐分配/释放（命名空间内可见）
// ===========================================================================

static inline void* aligned_malloc(size_t bytes) {
    size_t aligned_bytes = (bytes + 63) & ~63;
    return std::aligned_alloc(64, aligned_bytes);
}

static inline void aligned_free(void* ptr) {
    std::free(ptr);
}

static inline float* aligned_alloc_float(size_t n) {
    return (float*)aligned_malloc(n * sizeof(float));
}

// ===========================================================================
// AlignedBuffer — RAII 对齐缓冲区（每个命名空间独立实例）
// ===========================================================================

struct AlignedBuffer {
    float* ptr_ = nullptr;
    size_t cap_ = 0;

    AlignedBuffer() = default;
    explicit AlignedBuffer(size_t n) { resize(n); }
    ~AlignedBuffer() { aligned_free(ptr_); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&& o) noexcept : ptr_(o.ptr_), cap_(o.cap_) { o.ptr_ = nullptr; o.cap_ = 0; }
    AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
        if (this != &o) { aligned_free(ptr_); ptr_ = o.ptr_; cap_ = o.cap_; o.ptr_ = nullptr; o.cap_ = 0; }
        return *this;
    }

    void resize(size_t n) {
        if (n > cap_) { aligned_free(ptr_); ptr_ = aligned_alloc_float(n); cap_ = n; }
    }
    void zero() { if (ptr_) std::memset(ptr_, 0, cap_ * sizeof(float)); }
    float* data() { return ptr_; }
    const float* data() const { return ptr_; }
    size_t capacity() const { return cap_; }
    explicit operator bool() const { return ptr_ != nullptr; }
};

// ===========================================================================
// ThreadLocalBuffer — 线程本地缓冲区
// ===========================================================================

struct ThreadLocalBuffer {
    float* ptr = nullptr;
    size_t sz = 0;

    ThreadLocalBuffer() = default;
    ~ThreadLocalBuffer() { aligned_free(ptr); }

    float* get(size_t n) {
        if (n > sz) { aligned_free(ptr); ptr = aligned_alloc_float(n); sz = n; }
        return ptr;
    }

    ThreadLocalBuffer(const ThreadLocalBuffer&) = delete;
    ThreadLocalBuffer& operator=(const ThreadLocalBuffer&) = delete;
};

#define THREAD_LOCAL_BUFFER(name, size_expr) \
    thread_local ThreadLocalBuffer _tls_##name; \
    float* name = _tls_##name.get(size_expr)

static inline __m256 sigmoid256_ps(__m256 x)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one,
        _mm256_add_ps(one, exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), x))));
}

static inline __m256 tanh256_ps(__m256 x)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 e = exp256_ps(_mm256_add_ps(x, x));       /* e = exp(2x) */
    return _mm256_div_ps(_mm256_sub_ps(e, one), _mm256_add_ps(e, one));
}

/* in-place activations (scalar tail: hidden dims may not be ×8) */
static inline void sigmoid_inplace(float *x, int n)
{
    int n8 = n & ~7;
    for (int i = 0; i < n8; i += 8)
        _mm256_storeu_ps(x + i, sigmoid256_ps(_mm256_loadu_ps(x + i)));
    for (int i = n8; i < n; i++) x[i] = sigmoid(x[i]);
}

static inline void tanh_inplace(float *x, int n)
{
    int n8 = n & ~7;
    for (int i = 0; i < n8; i += 8)
        _mm256_storeu_ps(x + i, tanh256_ps(_mm256_loadu_ps(x + i)));
    for (int i = n8; i < n; i++) x[i] = tanhf(x[i]);
}

/* out[i] = sigmoid(a0[i] + x[i])  — a-gate (n = DIM, multiple of 8) */
static inline void a_gate(const float *a0, const float *x, float *out, int n)
{
    for (int i = 0; i < n; i += 8)
        _mm256_storeu_ps(out + i,
            sigmoid256_ps(_mm256_add_ps(_mm256_loadu_ps(a0 + i),
                                        _mm256_loadu_ps(x + i))));
}

/* ════════════════════════════════════════════════════════════════════
 * Weights — fp16-only storage
 * ════════════════════════════════════════════════════════════════════ */

/*
 * We store ONE fp16 buffer (data16).
 *
 * Big weights (matmul) stay as fp16:
 *   emb, head, R, K, V, O, w1, w2, a1, a2, g1, g2, v1, v2, ffn_K, ffn_V
 *
 * Tiny element-wise weights are converted to fp32 on load and cached:
 *   ln_w/b, mixer weights, w0/a0/v0, k_k/k_a, r_k, etc.
 *
 * Model dimensions are read from the binary header.
 */

struct Weights {
    /* ── model dimensions ───────────────────────────────────────── */
    int dim      = 0;
    int n_heads  = 0;
    int _h_size_   = 0;
    int n_layers = 0;
    int vocab    = 0;
    int ffn_dim  = 0;

    uint16_t *data16 = nullptr;  // fp16 raw buffer

    /* global weights — fp16 for on-the-fly conversion */
    const uint16_t *emb;     // [VOCAB, DIM]
    const uint16_t *head;    // [DIM, VOCAB]

    /* global element-wise — converted to fp32 */
    float *ln_out_w = nullptr;  // [DIM]
    float *ln_out_b = nullptr;  // [DIM]

    /* per-layer weights — a flat array (no fixed N_LAYERS) */
    struct Layer {
        /* Element-wise (converted to fp32 at load) */
        float *ln1_w, *ln1_b;
        float *ln2_w, *ln2_b;
        float *x_r, *x_w, *x_k, *x_v, *x_a, *x_g; // time-shift mixers
        float *w0, *a0, *v0;
        float *k_k, *k_a;
        float *r_k;
        float *ln_x_w, *ln_x_b;
        float *ffn_x_k;

        /* Bonus-path hidden dimensions (inferred from weight sizes at load) */
        int w1_hidden, a1_hidden, g1_hidden, v1_hidden;

        /* Matmul weights — fp16 (for both decode gemv + prefill gemm) */
        const uint16_t *R, *K, *V, *O;
        const uint16_t *w1, *w2, *a1, *a2;
        const uint16_t *g1, *g2, *v1, *v2;
        const uint16_t *ffn_K, *ffn_V;
    };
    std::vector<Layer> layers;

    /* Pool for small fp32 converted weights */
    float *ew_pool = nullptr;

    bool load(const char *path);
    ~Weights() { free(data16); free(ew_pool); }
};

/* ── Load 16-bit binary, convert small weights to fp32 ───────────── */
bool Weights::load(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return false;

    /* magic identifies the 16-bit storage format (must match build) */
#ifdef RWKV_WEIGHTS_BF16
    char magic[7];
    fread(magic, 1, 6, f);
    magic[6] = 0;
    if (strcmp(magic, "RWKV7B") != 0) { fclose(f); return false; }
#else
    char magic[6];
    fread(magic, 1, 5, f);
    magic[5] = 0;
    if (strcmp(magic, "RWKV7") != 0) { fclose(f); return false; }
#endif

    int header[7];
    fread(header, 4, 7, f);
    n_layers = header[0];
    dim      = header[1];
    n_heads  = header[2];
    // h_size   = header[3];
    vocab    = header[4];
    ffn_dim  = header[5];
    int n_blocks = header[6];

    printf("Model: %d layers, dim=%d, heads=%d, head_size=%d, vocab=%d, ffn_dim=%d\n",
           n_layers, dim, n_heads, h_size, vocab, ffn_dim);

    /* read offset table */
    std::vector<uint64_t> offsets(n_blocks), sizes(n_blocks);
    for (int i = 0; i < n_blocks; i++) {
        fread(&offsets[i], 8, 1, f);
        fread(&sizes[i],  8, 1, f);
    }

    /* read name table */
    std::vector<std::string> names(n_blocks);
    for (int i = 0; i < n_blocks; i++) {
        uint16_t len;
        fread(&len, 2, 1, f);
        names[i].resize(len);
        fread(&names[i][0], 1, len, f);
    }

    /* find contiguous data range */
    uint64_t min_off = offsets[0], max_end = 0;
    for (int i = 0; i < n_blocks; i++) {
        if (offsets[i] < min_off) min_off = offsets[i];
        max_end = std::max(max_end, offsets[i] + sizes[i] * 2);
    }
    uint64_t base = min_off;
    size_t   data_bytes = max_end - base;

    /* read fp16 raw data (single buffer) */
    uint16_t *raw = (uint16_t *)aligned_alloc(64, data_bytes);
    fseek(f, base, SEEK_SET);
    fread(raw, 1, data_bytes, f);
    fclose(f);

    data16 = raw;

    /* ── resolve fp16 pointers ──────────────────────────────────── */
    auto ptr16 = [&](const char *name) -> uint16_t * {
        for (int i = 0; i < n_blocks; i++) {
            if (names[i] == name) {
                size_t offset = 0;
                for (int j = 0; j < i; j++) offset += sizes[j];
                return raw + offset;
            }
        }
        return nullptr;
    };

    /* ── count element-wise elements for pool allocation ─────────── */
    size_t ew_count = 0;
    const char *ew_names[] = {
        "ln_out.weight","ln_out.bias",
        "L%d.ln1.w","L%d.ln1.b","L%d.ln2.w","L%d.ln2.b",
        "L%d.att.x_r","L%d.att.x_w","L%d.att.x_k","L%d.att.x_v",
        "L%d.att.x_a","L%d.att.x_g",
        "L%d.att.w0","L%d.att.a0","L%d.att.v0",
        "L%d.att.k_k","L%d.att.k_a","L%d.att.r_k",
        "L%d.att.ln_x.w","L%d.att.ln_x.b",
        "L%d.ffn.x_k",
    };
    for (const char *nm : ew_names) {
        if (nm[0] == 'L') {
            for (int l = 0; l < n_layers; l++) {
                char key[256];
                snprintf(key, sizeof(key), nm, l);
                uint16_t *p = ptr16(key);
                if (p) {
                    for (int b = 0; b < n_blocks; b++)
                        if (names[b] == key) { ew_count += sizes[b]; break; }
                }
            }
        } else {
            uint16_t *p = ptr16(nm);
            if (p) {
                for (int b = 0; b < n_blocks; b++)
                    if (names[b] == nm) { ew_count += sizes[b]; break; }
            }
        }
    }
    /* allocate pool */
    float *pool = (float *)aligned_alloc(64, ew_count * sizeof(float));
    ew_pool = pool;
    size_t pool_off = 0;

    /* helper: convert a block from fp16 to pool, return fp32 ptr */
    auto to_f32 = [&](const char *name, size_t expected = 0) -> float * {
        for (int b = 0; b < n_blocks; b++) {
            if (names[b] == name) {
                size_t n = sizes[b];
                if (expected && n != expected) {
                    fprintf(stderr, "weight %s: size %zu != expected %zu\n",
                            name, n, expected);
                    return nullptr;
                }
                float *dst = pool + pool_off;
                size_t elem_off = (offsets[b] - base) / 2;
                convert_16_to_f32(raw + elem_off, dst, n);
                pool_off += n;
                return dst;
            }
        }
        return nullptr;
    };

    /* ── global weights ──────────────────────────────────────────── */
    emb      = ptr16("emb.weight");
    head     = ptr16("head.weight");
    ln_out_w = to_f32("ln_out.weight");
    ln_out_b = to_f32("ln_out.bias");

    /* ── per-layer weights ───────────────────────────────────────── */
    layers.resize(n_layers);
    char key[256];
    for (int i = 0; i < n_layers; i++) {
        #define BIND(field, fmt) \
            snprintf(key, sizeof(key), fmt, i); \
            layers[i].field = to_f32(key)
        #define BIND16(field, fmt) \
            snprintf(key, sizeof(key), fmt, i); \
            layers[i].field = ptr16(key)

        BIND(ln1_w, "L%d.ln1.w");  BIND(ln1_b, "L%d.ln1.b");
        BIND(ln2_w, "L%d.ln2.w");  BIND(ln2_b, "L%d.ln2.b");

        BIND(x_r, "L%d.att.x_r");  BIND(x_w, "L%d.att.x_w");
        BIND(x_k, "L%d.att.x_k");  BIND(x_v, "L%d.att.x_v");
        BIND(x_a, "L%d.att.x_a");  BIND(x_g, "L%d.att.x_g");

        BIND(w0, "L%d.att.w0");    BIND(a0, "L%d.att.a0");
        BIND(v0, "L%d.att.v0");
        BIND(k_k, "L%d.att.k_k");  BIND(k_a, "L%d.att.k_a");
        BIND(r_k, "L%d.att.r_k");
        BIND(ln_x_w, "L%d.att.ln_x.w");  BIND(ln_x_b, "L%d.att.ln_x.b");
        BIND(ffn_x_k, "L%d.ffn.x_k");

        /* Matmul weights — fp16 */
        BIND16(R, "L%d.att.R");     BIND16(K, "L%d.att.K");
        BIND16(V, "L%d.att.Vmat");  BIND16(O, "L%d.att.O");
        BIND16(w1, "L%d.att.w1");   BIND16(w2, "L%d.att.w2");
        BIND16(a1, "L%d.att.a1");   BIND16(a2, "L%d.att.a2");
        BIND16(g1, "L%d.att.g1");   BIND16(g2, "L%d.att.g2");
        BIND16(v1, "L%d.att.v1");   BIND16(v2, "L%d.att.v2");
        BIND16(ffn_K, "L%d.ffn.K"); BIND16(ffn_V, "L%d.ffn.V");

        #undef BIND
        #undef BIND16
    }

    /* ── compute bonus-path hidden dimensions from block sizes ───── */
    for (int i = 0; i < n_layers; i++) {
        char kbuf[256];
        auto hid_of = [&](const char *fmt) -> int {
            snprintf(kbuf, sizeof(kbuf), fmt, i);
            for (int b = 0; b < n_blocks; b++) {
                if (names[b] == kbuf) return (int)sizes[b] / dim;
            }
            return 0;
        };
        layers[i].w1_hidden = hid_of("L%d.att.w1");
        layers[i].a1_hidden = hid_of("L%d.att.a1");
        layers[i].g1_hidden = hid_of("L%d.att.g1");
        layers[i].v1_hidden = hid_of("L%d.att.v1");
    }

#ifdef RWKV_WEIGHTS_BF16
    const char *wfmt = "bf16";
#else
    const char *wfmt = "fp16";
#endif
    double mb = data_bytes / 1024.0 / 1024.0;
    printf("Loaded %.1f MB %s + %.3f MB fp32 element-wise cache\n",
           mb, wfmt, ew_count * sizeof(float) / 1024.0 / 1024.0);
    return true;
}

/* ── State (per-token recurrent) — dynamically sized ────────────── */
struct State {
    int dim, n_heads, n_layers;

    float *att_x_prev;   // [n_layers][dim]
    float *att_state;    // [n_layers][n_heads][h_size][h_size]
    float *ffn_x_prev;   // [n_layers][dim]

    State(int d, int nh, int hs, int nl)
        : dim(d), n_heads(nh), n_layers(nl)
    {
        size_t sz_a = (size_t)nl * d;
        size_t sz_s = (size_t)nl * nh * hs * hs;
        size_t sz_f = (size_t)nl * d;
        att_x_prev = (float *)aligned_alloc(64, sz_a * sizeof(float));
        att_state  = (float *)aligned_alloc(64, sz_s * sizeof(float));
        ffn_x_prev = (float *)aligned_alloc(64, sz_f * sizeof(float));
        memset(att_x_prev, 0, sz_a * sizeof(float));
        memset(att_state,  0, sz_s * sizeof(float));
        memset(ffn_x_prev, 0, sz_f * sizeof(float));
    }
    ~State() { free(att_x_prev); free(att_state); free(ffn_x_prev); }
    void reset() {
        size_t sz_a = (size_t)n_layers * dim;
        size_t sz_s = (size_t)n_layers * n_heads * h_size * h_size;
        size_t sz_f = (size_t)n_layers * dim;
        memset(att_x_prev, 0, sz_a * sizeof(float));
        memset(att_state,  0, sz_s * sizeof(float));
        memset(ffn_x_prev, 0, sz_f * sizeof(float));
    }

    /* accessor helpers */
    float *att_x_prev_at(int l) { return att_x_prev + (size_t)l * dim; }
    float *ffn_x_prev_at(int l) { return ffn_x_prev + (size_t)l * dim; }
    float *att_state_at(int l)  { return att_state + (size_t)l * n_heads * h_size * h_size; }
};


static inline float sum256(const __m256& x) {
    __m128 low128  = _mm256_castps256_ps128(x);
    __m128 high128 = _mm256_extractf128_ps(x, 1);
    __m128 sum128  = _mm_add_ps(low128, high128);
    __m128 shuf    = _mm_movehl_ps(sum128, sum128);     
    sum128         = _mm_add_ps(sum128, shuf);
    shuf           = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(1, 1, 1, 1));
    sum128         = _mm_add_ps(sum128, shuf);
    return _mm_cvtss_f32(sum128);
}

/* ════════════════════════════════════════════════════════════════════
 * Kernel library
 * ════════════════════════════════════════════════════════════════════ */

/* ── layer norm: y = (x - mean) / sqrt(var + eps) * w + b ──────── */
#ifdef ABLATE_LAYER_NORM
static inline void layer_norm(const float *x, float *y, const float *w, const float *b, int D) {
    double sum = 0, sq = 0;
    for (int i = 0; i < D; i++) { sum += x[i]; sq += (double)x[i] * x[i]; }
    float mean = (float)(sum / D);
    float var  = (float)(sq / D) - mean * mean;
    float inv  = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < D; i++) y[i] = (x[i] - mean) * inv * w[i] + b[i];
}
#else
static inline void layer_norm(
    const float *x, float *y,
    const float *w, const float *b,
    int D)
{
    /* 1st pass: mean + sq_sum in one AVX2 pass */
    __m256 sum = _mm256_setzero_ps();
    __m256 sq_sum = _mm256_setzero_ps();
    for (int i = 0; i < D; i += 8) {
        __m256 x_i = _mm256_loadu_ps(x + i);
        sum = _mm256_add_ps(sum, x_i);
        sq_sum = _mm256_fmadd_ps(x_i, x_i, sq_sum);
    }
    float mean = sum256(sum) / D;
    float var_sum = sum256(sq_sum) / D - mean * mean;
    float inv_std = 1.0f / sqrtf(var_sum + 1e-5f);

    /* 3rd pass: normalize + affine (AVX2, 8-wide) */
    for (int i = 0; i < D; i += 8) {
        __m256 xi = _mm256_loadu_ps(x + i);
        __m256 mn = _mm256_set1_ps(mean);
        __m256 is = _mm256_set1_ps(inv_std);
        __m256 wi = _mm256_loadu_ps(w + i);
        __m256 bi = _mm256_loadu_ps(b + i);
        xi = _mm256_sub_ps(xi, mn);
        xi = _mm256_mul_ps(xi, is);
        xi = _mm256_fmadd_ps(xi, wi, bi);
        _mm256_storeu_ps(y + i, xi);
    }
}
#endif

/* ── group norm: normalize each head independently ──────────────── */
#ifdef ABLATE_GROUP_NORM
static inline void group_norm(const float *x, float *y, const float *w, const float *b,
                        int n_heads, int _h_size_) {
    for (int h = 0; h < n_heads; h++) {
        const float *xh = x + h * h_size;
        float       *yh = y + h * h_size;
        double sum = 0, sq = 0;
        for (int i = 0; i < h_size; i++) { sum += xh[i]; sq += (double)xh[i] * xh[i]; }
        float mean = (float)(sum / h_size);
        float var  = (float)(sq / h_size) - mean * mean;
        float inv  = 1.0f / sqrtf(var + 64e-5f);
        for (int i = 0; i < h_size; i++)
            yh[i] = (xh[i] - mean) * inv * w[h * h_size + i] + b[h * h_size + i];
    }
}
#else
static inline void group_norm(
    const float *x, float *y,
    const float *w, const float *b,
    int n_heads, int _h_size_)
{
    for (int h = 0; h < n_heads; h++) {
        const float *xh = x + h * h_size;
        float       *yh = y + h * h_size;

        /* 1st pass: sum + sq_sum in one AVX2 pass */
        __m256 sum = _mm256_setzero_ps();
        __m256 sq_sum = _mm256_setzero_ps();
        for (int i = 0; i < h_size; i += 8) {
            __m256 xi = _mm256_loadu_ps(xh + i);
            sum    = _mm256_add_ps(sum, xi);
            sq_sum = _mm256_fmadd_ps(xi, xi, sq_sum);
        }
        float mean    = sum256(sum) / h_size;
        float var_sum = sum256(sq_sum) / h_size - mean * mean;
        float inv_std = 1.0f / sqrtf(var_sum + 64e-5f);

        for (int i = 0; i < h_size; i += 8) {
            __m256 xi = _mm256_loadu_ps(xh + i);
            __m256 mn = _mm256_set1_ps(mean);
            __m256 is = _mm256_set1_ps(inv_std);
            __m256 wi = _mm256_loadu_ps(w + h * h_size + i);
            __m256 bi = _mm256_loadu_ps(b + h * h_size + i);

            xi = _mm256_sub_ps(xi, mn);
            xi = _mm256_mul_ps(xi, is);
            xi = _mm256_fmadd_ps(xi, wi, bi);
            _mm256_storeu_ps(yh + i, xi);
        }
    }
}
#endif

/* ── time-shift: out = x + xx * shift  (AVX2) ──────────────────── */
#ifdef ABLATE_TIME_SHIFT
static inline void time_shift(const float *x, const float *xx, const float *shift,
                               float *out, int D) {
    for (int i = 0; i < D; i++) out[i] = x[i] + xx[i] * shift[i];
}
#else
static inline void time_shift(
    const float *x, const float *xx,
    const float *shift, float *out,
    int D)
{
    for (int i = 0; i < D; i += 8) {
        __m256 xi  = _mm256_loadu_ps(x + i);
        __m256 xxi = _mm256_loadu_ps(xx + i);
        __m256 si  = _mm256_loadu_ps(shift + i);
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(xxi, si, xi));
    }
}
#endif

/* ── vector add / mul / relu²  (AVX2) ──────────────────────────── */
#ifdef ABLATE_VEC_ADD
static inline void vec_add(float *y, const float *x, int D) {
    for (int i = 0; i < D; i++) y[i] += x[i];
}
#else
static inline void vec_add(float *y, const float *x, int D)
{
    for (int i = 0; i < D; i += 8)
        _mm256_storeu_ps(y + i,
            _mm256_add_ps(_mm256_loadu_ps(y + i), _mm256_loadu_ps(x + i)));
}
#endif

#ifdef ABLATE_VEC_MUL
static inline void vec_mul(float *y, const float *g, int D) {
    for (int i = 0; i < D; i++) y[i] *= g[i];
}
#else
static inline void vec_mul(float *y, const float *g, int D)
{
    for (int i = 0; i < D; i += 8)
        _mm256_storeu_ps(y + i,
            _mm256_mul_ps(_mm256_loadu_ps(y + i), _mm256_loadu_ps(g + i)));
}
#endif

#ifdef ABLATE_RELU_SQ
static inline void relu_sq(float *y, const float *x, int D) {
    for (int i = 0; i < D; i++) { float v = x[i] > 0 ? x[i] : 0; y[i] = v * v; }
}
#else
static inline void relu_sq(float *y, const float *x, int D)
{
    for (int i = 0; i < D; i += 8) {
        __m256 xi = _mm256_max_ps(_mm256_loadu_ps(x + i), _mm256_setzero_ps());
        _mm256_storeu_ps(y + i, _mm256_mul_ps(xi, xi));
    }
}
#endif

/* ── batched matmul via fp16 GEMM library ────────────────────────── */
static inline void matmul_batch(
    const float *X, const uint16_t *W, float *Y,
    int B, int M, int N)
{
    if (B <= 8) {
        // R/K/V projections alias their inputs; batched GEMV writes directly.
        AlignedBuffer result((size_t)B * N);
        gemv_f16_batchT(X, W, result.data(), B, M, N);
        memcpy(Y, result.data(), (size_t)B * N * sizeof(float));
        return;
    }
    gemm<float, unsigned short, float>((size_t)B, (size_t)N, (size_t)M,
                                        X, W, Y, false, false);
}

/* ── single-row matmul (for final head projection) ──────────────── */
static inline void matmul_1(
    const float *x, const uint16_t *W, float *y,
    int M, int N)
{
    gemv_f16(x, W, y, M, N);
}

/* ── K normalization + modification ─────────────────────────────── */
#ifdef ABLATE_K_NORMALIZE
static inline void k_normalize(
    float *const k, const float *const a, float *const kk,
    const float *const k_k, const float *const k_a,
    int T, int dim, int n_heads, int _h_size_)
{
    for (int t = 0; t < T; t++) {
        float *kt  = k  + t * dim;
        const float *at  = a  + t * dim;
        float *kkt = kk + t * dim;
        for (int h = 0; h < n_heads; h++) {
            double sum_sq = 0;
            for (int j = 0; j < h_size; j++) {
                int idx = h * h_size + j;
                float val = kt[idx] * k_k[idx];
                kkt[idx] = val;
                sum_sq += (double)val * val;
            }
            float inv = 1.0 / sqrt(sum_sq);
            for (int j = 0; j < h_size; j++)
                kkt[h * h_size + j] *= inv;
        }
        for (int i = 0; i < dim; i++)
            kt[i] = kt[i] * (1.0f + (at[i] - 1.0f) * k_a[i]);
    }
}
#else
static inline void k_normalize(
    float       *const k,      // [T, DIM]  (in/out)
    const float *const a,      // [T, DIM]
    float       *const kk,     // [T, DIM]  (out)
    const float *const k_k,    // [DIM]
    const float *const k_a,    // [DIM]
    int T, int dim, int n_heads, int _h_size_)
{
    for (int t = 0; t < T; t++) {
        float       *kt  = k  + t * dim;
        const float *at  = a  + t * dim;
        float       *kkt = kk + t * dim;

        /* kk = L2-normalize(k * k_k) per head */

        for (int h = 0; h < n_heads; h++) {
            float sum_sq = 0;
            for (int j = 0; j < h_size; j += 8) {
                int idx = h * h_size + j;
                __m256 k8 = _mm256_mul_ps(_mm256_loadu_ps(kt + idx), _mm256_loadu_ps(k_k + idx));
                _mm256_storeu_ps(kkt + idx, k8);
                k8 = _mm256_mul_ps(k8, k8);
                sum_sq   += sum256(k8);
            }
            __m256 inv = _mm256_set1_ps(1.0f / sqrtf(sum_sq));
            __m256 i1 = _mm256_set1_ps(1.0f);
            for (int j = 0; j < h_size; j += 8){
                int idx = h * h_size + j;
                __m256 _x = _mm256_loadu_ps(kkt + idx);
                _x = _mm256_mul_ps(_x, inv);
                _mm256_storeu_ps(kkt + idx, _x);
                _x = _mm256_loadu_ps(at + idx);
                _x = _mm256_fmadd_ps(_mm256_sub_ps(_x, i1), _mm256_loadu_ps(k_a + idx), i1);
                _mm256_storeu_ps(kt + idx, _mm256_mul_ps(_mm256_loadu_ps(kt + idx), _x));
            }
        }
    }
}
#endif

/* ── WKV state kernel (decode: T=1) ─────────────────────────────── */
#ifdef ABLATE_WKV
static inline void wkv_decode(
    const float *r, const float *w, const float *k_, const float *v,
    const float *kk, const float *a, const float *w0,
    float *state, float *y, int dim, int n_heads, int _h_size_)
{
    (void)dim;
    for (int h = 0; h < n_heads; h++) {
        float *S   = state + h * h_size * h_size;
        int    off = h * h_size;
        float w_val[64], kka[64];
        for (int i = 0; i < h_size; i++) {
            w_val[i] = expf(-0.606531f * sigmoid(w0[off + i] + w[off + i]));
            kka[i]   = kk[off + i] * a[off + i];
        }
        for (int i = 0; i < h_size; i++) {
            float *Si = S + i * h_size;
            float  vi = v[off + i];
            float sa = 0;
            for (int j = 0; j < h_size; j++) sa -= Si[j] * kk[off + j];
            float result = 0;
            for (int jj = 0; jj < h_size; jj++) {
                float s = Si[jj] * w_val[jj] + vi * k_[off + jj] + sa * kka[jj];
                Si[jj]  = s;
                result += s * r[off + jj];
            }
            y[off + i] = result;
        }
    }
}
#else
static inline void wkv_decode(
    const float *r,       // [DIM]
    const float *w,       // [DIM]
    const float *k,       // [DIM]
    const float *v,       // [DIM]
    const float *kk,      // [DIM]
    const float *a,       // [DIM]
    const float *w0,      // [DIM]
    float       *state,   // [N_HEADS][H_SIZE][H_SIZE]
    float       *y,       // [DIM]
    int dim, int n_heads, int _h_size_)
{
    (void)dim;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < n_heads; h++) {
        float *S    = state + h * h_size * h_size;
        int    off  = h * h_size;

        /* w_val = exp(-0.606531 * sigmoid(w0 + w)) */
        float w_val[64];  // max h_size is 64
        float kka[64];
        for (int i = 0; i < h_size; i++) {
            w_val[i] = expf(-0.606531f * sigmoid(w0[off + i] + w[off + i]));
            kka[i]   = kk[off + i] * a[off + i];
        }

        for (int i = 0; i < h_size; i++) {
            float *Si = S + i * h_size;
            float  vi = v[off + i];

            /* sa = row_i(S) · (-kk)   — AVX2 dot product */
            __m256 sa8 = _mm256_setzero_ps();
            for (int j = 0; j < h_size; j += 8) {
                __m256 sj = _mm256_loadu_ps(Si + j);
                __m256 kj = _mm256_loadu_ps(kk + off + j);
                sa8 = _mm256_fmadd_ps(sj, kj, sa8);
            }
            float sa = -sum256(sa8);

            /* S += S * w + v ⊗ k + sa ⊗ kka   ;   y = S · r */
            float result = 0;
            for (int jj = 0; jj < h_size; jj++) {
                float s = Si[jj] * w_val[jj] + vi * k[off + jj] + sa * kka[jj];
                Si[jj]  = s;
                result += s * r[off + jj];
            }
            y[off + i] = result;
        }
    }
}
#endif

/* ── WKV state kernel (prefill: T tokens) ───────────────────────── */
#ifdef ABLATE_WKV
static inline void wkv_prefill(
    const float *const r, const float *const w_, const float *const k_,
    const float *const v_, const float *const kk, const float *const a,
    const float *const w0, float *const state, float *const y,
    int T, int dim, int n_heads, int _h_size_)
{
    (void)dim;
    for (int h = 0; h < n_heads; h++) {
        float *S   = state + h * h_size * h_size;
        int    off = h * h_size;
        float w0_loc[64];
        for (int i = 0; i < h_size; i++) w0_loc[i] = w0[off + i];
        for (int t = 0; t < T; t++) {
            const float *rt  = r   + t * dim;
            const float *wt  = w_  + t * dim;
            const float *kt  = k_  + t * dim;
            const float *vt  = v_  + t * dim;
            const float *kkt = kk + t * dim;
            const float *at  = a  + t * dim;
            float w_val[64], kka[64];
            for (int i = 0; i < h_size; i++) {
                w_val[i] = expf(-0.606531f * sigmoid(w0_loc[i] + wt[off + i]));
                kka[i]   = kkt[off + i] * at[off + i];
            }
            float *yt = y + t * dim;
            for (int i = 0; i < h_size; i++) {
                float *Si = S + i * h_size;
                float  vi = vt[off + i];
                float sa = 0;
                for (int j = 0; j < h_size; j++) sa -= Si[j] * kkt[off + j];
                float result = 0;
                for (int jj = 0; jj < h_size; jj++) {
                    float s = Si[jj] * w_val[jj] + vi * kt[off + jj] + sa * kka[jj];
                    Si[jj]  = s;
                    result += s * rt[off + jj];
                }
                yt[off + i] = result;
            }
        }
    }
}
#else
static inline void wkv_prefill(
    const float *const r,       // [T, DIM]
    const float *const w,       // [T, DIM]
    const float *const k,       // [T, DIM]
    const float *const v,       // [T, DIM]
    const float *const kk,      // [T, DIM]
    const float *const a,       // [T, DIM]
    const float *const w0,      // [DIM]
    float       *const state,   // [N_HEADS][H_SIZE][H_SIZE]
    float       *const y,       // [T, DIM]
    int T, int dim, int n_heads, int _h_size_)
{
    (void)dim;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < n_heads; h++) {
        float *S   = state + h * h_size * h_size;
        int    off = h * h_size;

        /* copy w0 locally (same for all T) */
        float w0_loc[64];
        for (int i = 0; i < h_size; i++)
            w0_loc[i] = w0[off + i];

        for (int t = 0; t < T; t++) {
            const float *const rt  = r  + t * dim;
            const float *const wt  = w  + t * dim;
            const float *const kt  = k  + t * dim;
            const float *const vt  = v  + t * dim;
            const float *const kkt = kk + t * dim;
            const float *const at  = a  + t * dim;

            float w_val[64], kka[64];
            for (int i = 0; i < h_size; i++) {
                w_val[i] = expf(-0.606531f * sigmoid(w0_loc[i] + wt[off + i]));
                kka[i]   = kkt[off + i] * at[off + i];
            }

            float *const yt = y + t * dim;
            for (int i = 0; i < h_size; i++) {
                float *const Si = S + i * h_size;
                const float  vi = vt[off + i];

                /* sa = row_i(S) · (-kk)  — AVX2 dot product */
                __m256 sa8 = _mm256_setzero_ps();
                for (int j = 0; j < h_size; j += 8) {
                    __m256 sj = _mm256_loadu_ps(Si + j);
                    __m256 kj = _mm256_loadu_ps(kkt + off + j);
                    sa8 = _mm256_fmadd_ps(sj, kj, sa8);
                }
                float sa = -sum256(sa8);

                /* S update + result */
                float result = 0;
                for (int jj = 0; jj < h_size; jj++) {
                    float s = Si[jj] * w_val[jj] + vi * kt[off + jj] + sa * kka[jj];
                    Si[jj]  = s;
                    result += s * rt[off + jj];
                }
                yt[off + i] = result;
            }
        }
    }
}
#endif

/* ════════════════════════════════════════════════════════════════════
 * Forward pass — single token (decode)  — fp16 GEMV path
 * ════════════════════════════════════════════════════════════════════ */

void forward_one(
    const Weights &W, State &S,
    int token, float *const logits)
{
    int dim      = W.dim;
    int n_heads  = W.n_heads;
    int n_layers = W.n_layers;
    int vocab    = W.vocab;
    int ffn_dim  = W.ffn_dim;

    /* buffer layout (each C = DIM floats):
     *   [0]: x      [1]: xx    [2]: r     [3]: wb
     *   [4]: kb     [5]: vb    [6]: ab    [7]: gb
     *   [8]: kkb    [9]: t1    [10]: xv
     *   [11]: r_out [12]: k_out [13]: v_out [14]: wv_tmp
     *   [15]: ftmp (FFN_DIM floats)
     *   [16]: v_first
     */
    static float* scratch_buf = nullptr;
    if (!scratch_buf) scratch_buf = (float*)aligned_alloc(64, (15 * dim + ffn_dim + dim) * 4);

    float *const x      = scratch_buf;
    float *const xx     = scratch_buf + 1 * dim;
    float *const r      = scratch_buf + 2 * dim;
    float *const wb     = scratch_buf + 3 * dim;
    float *const kb     = scratch_buf + 4 * dim;
    float *const vb     = scratch_buf + 5 * dim;
    float *const ab     = scratch_buf + 6 * dim;
    float *const gb     = scratch_buf + 7 * dim;
    float *const kkb    = scratch_buf + 8 * dim;
    float *const t1     = scratch_buf + 9 * dim;
    float *const xv     = scratch_buf + 10 * dim;
    float *const r_out  = scratch_buf + 11 * dim;
    float *const k_out  = scratch_buf + 12 * dim;
    float *const v_out  = scratch_buf + 13 * dim;
    float *const wv_tmp = scratch_buf + 14 * dim;
    float *const ftmp   = scratch_buf + 15 * dim;   /* FFN_DIM floats */
    float *const v_first = scratch_buf + 15 * dim + ffn_dim;

    /* embedding: 16-bit → fp32 on the fly (compile-time format) */
    for (int i = 0; i < dim; i += 8)
        _mm256_storeu_ps(x + i, load_16x8(W.emb + token * dim + i));

    for (int l = 0; l < n_layers; l++) {
        const Weights::Layer &L = W.layers[l];

        /* ── LN1 + time-shift ──────────────────────────────────── */
        layer_norm(x, xx, L.ln1_w, L.ln1_b, dim);

        float *axp = S.att_x_prev_at(l);

        /* SIMD: t1 = prev - xx;  update prev = xx */
        for (int i = 0; i < dim; i += 8) {
            __m256 xx_i   = _mm256_loadu_ps(xx + i);
            __m256 prev_i = _mm256_loadu_ps(axp + i);
            __m256 t1_i   = _mm256_sub_ps(prev_i, xx_i);
            _mm256_storeu_ps(t1 + i, t1_i);
            _mm256_storeu_ps(axp + i, xx_i);
        }

        time_shift(xx, t1, L.x_v, vb, dim);

        // Small models benefit from independent projection tasks. Large matrices
        // need all cores streaming one projection together (no nested teams).
        auto project = [dim](const float *in, const uint16_t *weight,
                             float *out, int m, int n) {
            if (dim <= 1024) gemv_f16_st(in, weight, out, m, n);
            else gemv_f16(in, weight, out, m, n);
        };
        #pragma omp parallel sections if(dim <= 1024)
        {
            #pragma omp section
            {
                /* stack-local scratch sized for largest bonus path */
                float tmp[1024];  /* max(g1=480 for 7B, w1=128, a1=128, v1=96) */

                time_shift(xx, t1, L.x_w, wb, dim);
                time_shift(xx, t1, L.x_a, ab, dim);
                time_shift(xx, t1, L.x_g, gb, dim);

                /* ── w = tanh(xw @ w1) @ w2 ─────────────────── */
                /* Bonus-path hidden dims are stored in the Layer struct at load time */
                int w1_hidden = L.w1_hidden;
                int a1_hidden = L.a1_hidden;
                int g1_hidden = L.g1_hidden;
                int v1_hidden = L.v1_hidden;

                project(wb, L.w1, tmp, dim, w1_hidden);
                for (int i = 0; i < w1_hidden; i++) tmp[i] = tanhf(tmp[i]);
                project(tmp, L.w2, wb, w1_hidden, dim);

                /* ── a = sigmoid(a0 + (xa @ a1) @ a2) ────────── */
                project(ab, L.a1, tmp, dim, a1_hidden);
                project(tmp, L.a2, wv_tmp, a1_hidden, dim);
                for (int i = 0; i < dim; i++)
                    ab[i] = sigmoid(L.a0[i] + wv_tmp[i]);

                /* ── g = sigmoid(xg @ g1) @ g2 ──────────────── */
                project(gb, L.g1, tmp, dim, g1_hidden);
                for (int i = 0; i < g1_hidden; i++) tmp[i] = sigmoid(tmp[i]);
                project(tmp, L.g2, gb, g1_hidden, dim);

                /* ── v0 gate: (vb @ v1) @ v2 → xv  (safe: all sections only read vb) ── */
                if (l > 0) {
                    project(vb, L.v1, tmp, dim, v1_hidden);
                    project(tmp, L.v2, xv, v1_hidden, dim);
                }
            }

            #pragma omp section
            {
                time_shift(xx, t1, L.x_r, r,  dim);
                project(r, L.R, r_out, dim, dim);
            }

            #pragma omp section
            {
                time_shift(xx, t1, L.x_k, kb, dim);
                project(kb, L.K, k_out, dim, dim);
            }

            #pragma omp section
            {
                project(vb, L.V, v_out, dim, dim);
            }
        }
        /* ── K normalize (uses post-projection buffers directly) ── */
        k_normalize(k_out, ab, kkb, L.k_k, L.k_a, 1, dim, n_heads, h_size);

        /* ── v0 adjustment ──────────────────────────────────────── */
        if (l == 0) {
            memcpy(v_first, v_out, dim * sizeof(float));
        } else {
            /* xv already holds pre-computed (vb @ v1) @ v2 */
            for (int i = 0; i < dim; i++)
                v_out[i] += (v_first[i] - v_out[i]) * sigmoid(L.v0[i] + xv[i]);
        }

        /* ── WKV ────────────────────────────────────────────────── */
        wkv_decode(r_out, wb, k_out, v_out, kkb, ab, L.w0,
                   S.att_state_at(l), xx, dim, n_heads, h_size);

        /* ── group norm + bonus term + output ───────────────────── */
        /* xv is free after v0 adjustment — reuse for attention output */
        group_norm(xx, xv, L.ln_x_w, L.ln_x_b, n_heads, h_size);

        for (int h = 0; h < n_heads; h++) {
            float bonus = 0;
            for (int j = 0; j < h_size; j++) {
                int idx = h * h_size + j;
                bonus += r_out[idx] * k_out[idx] * L.r_k[idx];
            }
            for (int j = 0; j < h_size; j++)
                xv[h * h_size + j] += bonus * v_out[h * h_size + j];
        }

        vec_mul(xv, gb, dim);
        gemv_f16(xv, L.O, xx, dim, dim);
        vec_add(x, xx, dim);

        /* ── LN2 + channel mix (FFN) with sparse kernel ─────────── */
        layer_norm(x, xx, L.ln2_w, L.ln2_b, dim);

        float *fxp = S.ffn_x_prev_at(l);

        /* SIMD: t1 = prev - xx;  update prev = xx */
        for (int i = 0; i < dim; i += 8) {
            __m256 xx_i   = _mm256_loadu_ps(xx + i);
            __m256 prev_i = _mm256_loadu_ps(fxp + i);
            __m256 t1_i   = _mm256_sub_ps(prev_i, xx_i);
            _mm256_storeu_ps(t1 + i, t1_i);
            _mm256_storeu_ps(fxp + i, xx_i);
        }
        time_shift(xx, t1, L.ffn_x_k, k_out, dim);

        /* k = relu(x @ ffn_K)^2  →  sparse GEMV: k @ ffn_V */
        gemv_f16(k_out, L.ffn_K, ftmp, dim, ffn_dim);
        relu_sq(ftmp, ftmp, ffn_dim);

        /* sparse FFN: only non-zero k entries participate */
        memset(xv, 0, dim * sizeof(float));
        gemv_f16_sparse(ftmp, L.ffn_V, xv, ffn_dim, dim);
        vec_add(x, xv, dim);

    }

    /* ── final LN + head ─────────────────────────────────────── */
    layer_norm(x, xx, W.ln_out_w, W.ln_out_b, dim);
    gemv_f16(xx, W.head, logits, dim, vocab);
}

/* ════════════════════════════════════════════════════════════════════
 * Forward pass — sequence (prefill)  — fp16 gemm
 * ════════════════════════════════════════════════════════════════════ */

void forward_seq(
    const Weights &W, State &S,
    const int *tokens, int T,
    float *const logits)
{
    int dim      = W.dim;
    int n_heads  = W.n_heads;
    int n_layers = W.n_layers;
    int vocab    = W.vocab;
    int ffn_dim  = W.ffn_dim;

    int TC = T * dim;
    int TF = T * ffn_dim;

    /* buffer layout:
     *   0:  x      1:  r      2:  wb     3:  kb     4:  vb_
     *   5:  ab     6:  gb     7:  kkb    8:  tmp    9:  tmp2
     *   10: xv_s   11: ftmp (TF)   11+TF: ftmp2 (TF)
     *   11+2*TF: v_first (TC)
     */
    static AlignedBuffer buf_storage;
    size_t buf_size = 11 * TC + 2 * TF + TC;  // 12 * TC + 2 * TF
    buf_storage.resize(buf_size);
    float *const buf = buf_storage.data();

    float *const x      = buf;
    float *const r      = buf + 1 * TC;
    float *const wb     = buf + 2 * TC;
    float *const kb     = buf + 3 * TC;
    float *const vb_    = buf + 4 * TC;
    float *const ab     = buf + 5 * TC;
    float *const gb     = buf + 6 * TC;
    float *const kkb    = buf + 7 * TC;
    float *const tmp    = buf + 8 * TC;
    float *const tmp2   = buf + 9 * TC;
    float *const xv_s   = buf + 10 * TC;
    float *const ftmp   = buf + 11 * TC;
    float *const ftmp2  = buf + 11 * TC + TF;
    float *const v_first = buf + 11 * TC + 2 * TF;

    /* embedding: 16-bit → fp32 on the fly (compile-time format) */
    for (int t = 0; t < T; t++) {
        float *__restrict const xt = x + t * dim;
        const uint16_t *__restrict const emb_t = W.emb + tokens[t] * dim;
        for (int i = 0; i < dim; i += 8)
            _mm256_storeu_ps(xt + i, load_16x8(emb_t + i));
    }

    for (int l = 0; l < n_layers; l++) {
        const Weights::Layer &L = W.layers[l];

        /* ── LN1 (per token) ────────────────────────────────────── */
        for (int t = 0; t < T; t++)
            layer_norm(x + t * dim, tmp + t * dim, L.ln1_w, L.ln1_b, dim);

        /* ── time-shift: xx[0] = x_prev - x[0], xx[t] = x[t-1] - x[t] ─ */
        {
            float *__restrict const dst0 = tmp2;
            const float *__restrict const now0 = tmp;
            const float *__restrict const prev_s = S.att_x_prev_at(l);
            for (int i = 0; i < dim; i += 8) {
                __m256 p = _mm256_loadu_ps(prev_s + i);
                __m256 n = _mm256_loadu_ps(now0 + i);
                _mm256_storeu_ps(dst0 + i, _mm256_sub_ps(p, n));
            }
        }
        for (int t = 1; t < T; t++) {
            const float *__restrict const prev_tok = tmp  + (t - 1) * dim;
            const float *__restrict const cur_tok  = tmp  + t * dim;
            float       *__restrict const dst      = tmp2 + t * dim;
            for (int i = 0; i < dim; i += 8) {
                __m256 p = _mm256_loadu_ps(prev_tok + i);
                __m256 c = _mm256_loadu_ps(cur_tok + i);
                _mm256_storeu_ps(dst + i, _mm256_sub_ps(p, c));
            }
        }
        {
            const float *__restrict const last = tmp + (T - 1) * dim;
            float *__restrict const prev_dst = S.att_x_prev_at(l);
            for (int i = 0; i < dim; i += 8)
                _mm256_storeu_ps(prev_dst + i, _mm256_loadu_ps(last + i));
        }

        /* ── shifted inputs ─────────────────────────────────────── */
        for (int t = 0; t < T; t++) {
            const float *const xt  = tmp  + t * dim;
            const float *const xxt = tmp2 + t * dim;
            time_shift(xt, xxt, L.x_r, r   + t * dim, dim);
            time_shift(xt, xxt, L.x_w, wb  + t * dim, dim);
            time_shift(xt, xxt, L.x_k, kb  + t * dim, dim);
            time_shift(xt, xxt, L.x_v, vb_ + t * dim, dim);
            time_shift(xt, xxt, L.x_a, ab  + t * dim, dim);
            time_shift(xt, xxt, L.x_g, gb  + t * dim, dim);
        }
        /* ── pre-compute v0 gate into xv_s (before V-projection overwrites vb_) ── */
        if (l > 0) {
            int v1_hidden = L.v1_hidden;

            matmul_batch(vb_, L.v1, tmp, T, dim, v1_hidden);
            matmul_batch(tmp, L.v2, xv_s, T, v1_hidden, dim);
        }

        /* ── batched matmuls (fp16 weights via gemm library) ────── */
        matmul_batch(r,   L.R, r,   T, dim, dim);
        matmul_batch(kb,  L.K, kb,  T, dim, dim);
        matmul_batch(vb_, L.V, vb_, T, dim, dim);

        /* w */
        {
            int w1_hidden = L.w1_hidden;

            matmul_batch(wb, L.w1, tmp, T, dim, w1_hidden);
            for (int i = 0; i < T * w1_hidden; i++) tmp[i] = tanhf(tmp[i]);
            matmul_batch(tmp, L.w2, wb, T, w1_hidden, dim);
        }

        /* a */
        {
            int a1_hidden = L.a1_hidden;

            matmul_batch(ab, L.a1, tmp, T, dim, a1_hidden);
            matmul_batch(tmp, L.a2, tmp2, T, a1_hidden, dim);
            for (int t = 0; t < T; t++)
                for (int i = 0; i < dim; i++)
                    ab[t * dim + i] = sigmoid(L.a0[i] + tmp2[t * dim + i]);
        }

        /* g */
        {
            int g1_hidden = L.g1_hidden;

            matmul_batch(gb, L.g1, tmp, T, dim, g1_hidden);
            for (int i = 0; i < T * g1_hidden; i++) tmp[i] = sigmoid(tmp[i]);
            matmul_batch(tmp, L.g2, gb, T, g1_hidden, dim);
        }

        /* ── K normalize ────────────────────────────────────────── */
        k_normalize(kb, ab, kkb, L.k_k, L.k_a, T, dim, n_heads, h_size);

        /* ── v0 adjustment ──────────────────────────────────────── */
        if (l == 0) {
            memcpy(v_first, vb_, TC * sizeof(float));
        } else {
            for (int t = 0; t < T; t++) {
                float *const vt = vb_ + t * dim;
                const float *const vf = v_first + t * dim;
                const float *const res = xv_s + t * dim;
                for (int i = 0; i < dim; i++)
                    vt[i] += (vf[i] - vt[i]) * sigmoid(L.v0[i] + res[i]);
            }
        }

        /* ── WKV (prefill) ──────────────────────────────────────── */
        wkv_prefill(r, wb, kb, vb_, kkb, ab, L.w0,
                    S.att_state_at(l), tmp, T, dim, n_heads, h_size);

        /* ── group norm + bonus + output ────────────────────────── */
        /* Every layer needs each token's FFN input for the next time shift. */
        for (int t = 0; t < T; t++)
            group_norm(tmp + t * dim, tmp2 + t * dim,
                       L.ln_x_w, L.ln_x_b, n_heads, h_size);

        for (int t = 0; t < T; t++) {
            float       *const yt = tmp2 + t * dim;
            const float *const rt = r    + t * dim;
            const float *const kt = kb   + t * dim;
            const float *const vt = vb_  + t * dim;
            for (int h = 0; h < n_heads; h++) {
                float bonus = 0;
                for (int j = 0; j < h_size; j++) {
                    int idx = h * h_size + j;
                    bonus += rt[idx] * kt[idx] * L.r_k[idx];
                }
                for (int j = 0; j < h_size; j++)
                    yt[h * h_size + j] += bonus * vt[h * h_size + j];
            }
        }

        for (int t = 0; t < T; t++)
            vec_mul(tmp2 + t * dim, gb + t * dim, dim);
        matmul_batch(tmp2, L.O, tmp, T, dim, dim);
        for (int t = 0; t < T; t++)
            vec_add(x + t * dim, tmp + t * dim, dim);

        // Only the final FFN output is consumed in last-logit prefill.
        // Its time shift still needs the preceding LN2 row; recurrent FFN state
        // is the final LN2 input, independent of either FFN projection.
        const bool last_layer = l == n_layers - 1;
        const int first = last_layer ? std::max(0, T - 2) : 0;
        /* ── LN2 + channel mix ──────────────────────────────── */
        for (int t = first; t < T; t++)
            layer_norm(x + t * dim, tmp + t * dim, L.ln2_w, L.ln2_b, dim);

        for (int t = first; t < T; t++) {
            float *const xt = tmp  + t * dim;
            float *const d  = tmp2 + t * dim;
            float *const fxp = S.ffn_x_prev_at(l);
            for (int i = 0; i < dim; i++) {
                d[i] = fxp[i] - xt[i];
                fxp[i] = xt[i];
            }
            time_shift(xt, d, L.ffn_x_k, kb + t * dim, dim);
        }

        if (last_layer) {
            gemv_f16(kb + (T - 1) * dim, L.ffn_K, ftmp, dim, ffn_dim);
            relu_sq(ftmp, ftmp, ffn_dim);
            gemv_f16_sparse(ftmp, L.ffn_V, x + (T - 1) * dim, ffn_dim, dim);
        } else {
            matmul_batch(kb, L.ffn_K, ftmp, T, dim, ffn_dim);
            relu_sq(ftmp2, ftmp, TF);
            matmul_batch(ftmp2, L.ffn_V, tmp, T, ffn_dim, dim);
            for (int t = 0; t < T; t++)
                vec_add(x + t * dim, tmp + t * dim, dim);
        }
    }

    /* ── final LN + head (last token only) ────────────────────── */
    {
        float *const x_last  = x   + (T - 1) * dim;
        float *const xx_last = tmp + (T - 1) * dim;  /* reuse tmp for LN output */
        layer_norm(x_last, xx_last, W.ln_out_w, W.ln_out_b, dim);
        matmul_1(xx_last, W.head, logits, dim, vocab);
    }
}

/* ════════════════════════════════════════════════════════════════════
 * Forward pass — speculative-decode verification window (T ≤ 8)
 * ════════════════════════════════════════════════════════════════════
 *
 * Batched forward over a short token window. For EVERY token the full
 * logits vector and (optionally) a snapshot of the complete recurrent
 * state are produced, so a speculative verifier can accept any prefix
 * and roll back to the matching state.
 *
 * All batched projections go through gemv_f16_batchT: with T ≤ 8 the
 * accumulators stay in YMM registers and each 16-bit weight matrix is
 * streamed exactly once (no GEMM packing, 1× weight traffic).
 *
 * Snapshot layout per token (floats), matches State exactly:
 *   [0]           att_x_prev [n_layers * dim]
 *   [nl*dim]      att_state  [n_layers * n_heads * h_size²]
 *   [nl*dim+ns]   ffn_x_prev [n_layers * dim]
 * i.e. (h_size²+2)/64·... — total (2*dim + n_heads*h_size²) * n_layers
 * = rwkv_state_size(w). The caller pre-allocates T * state_size floats.
 */

static inline size_t state_snapshot_size(int nl, int dim, int nh)
{
    return (size_t)nl * (2 * (size_t)dim + (size_t)nh * h_size * h_size);
}

static inline void state_save(const State &S, float *dst)
{
    size_t na = (size_t)S.n_layers * S.dim;
    size_t ns = (size_t)S.n_layers * S.n_heads * h_size * h_size;
    memcpy(dst,               S.att_x_prev, na * sizeof(float));
    memcpy(dst + na,          S.att_state,  ns * sizeof(float));
    memcpy(dst + na + ns,     S.ffn_x_prev, na * sizeof(float));
}

static inline void state_restore(State &S, const float *src)
{
    size_t na = (size_t)S.n_layers * S.dim;
    size_t ns = (size_t)S.n_layers * S.n_heads * h_size * h_size;
    memcpy(S.att_x_prev, src,               na * sizeof(float));
    memcpy(S.att_state,  src + na,          ns * sizeof(float));
    memcpy(S.ffn_x_prev, src + na + ns,     na * sizeof(float));
}

/* ── WKV window kernel: T ≤ 8 tokens, AVX2-fused inner loop,
 *    optional per-token state snapshots (snap may be NULL) ────────── */
static inline void wkv_window(
    const float *const r, const float *const w_, const float *const k_,
    const float *const v_, const float *const kk, const float *const a,
    const float *const w0, float *const state, float *const y,
    int T, int dim, int n_heads, int _h_size_,
    float *const snap, size_t snap_stride)
{
    (void)dim;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < n_heads; h++) {
        float *S   = state + h * h_size * h_size;
        int    off = h * h_size;

        float w0_loc[64];
        for (int i = 0; i < h_size; i++) w0_loc[i] = w0[off + i];

        for (int t = 0; t < T; t++) {
            const float *const rt  = r   + t * dim;
            const float *const wt  = w_  + t * dim;
            const float *const kt  = k_  + t * dim;
            const float *const vt  = v_  + t * dim;
            const float *const kkt = kk  + t * dim;
            const float *const at  = a   + t * dim;

            float w_val[64], kka[64];
            for (int i = 0; i < h_size; i++) {
                w_val[i] = expf(-0.606531f * sigmoid(w0_loc[i] + wt[off + i]));
                kka[i]   = kkt[off + i] * at[off + i];
            }

            float *const yt = y + t * dim;
            for (int i = 0; i < h_size; i++) {
                float *const Si = S + i * h_size;
                const float  vi = vt[off + i];

                /* sa = row_i(S) · (-kk) */
                __m256 sa8 = _mm256_setzero_ps();
                for (int j = 0; j < h_size; j += 8) {
                    __m256 sj = _mm256_loadu_ps(Si + j);
                    __m256 kj = _mm256_loadu_ps(kkt + off + j);
                    sa8 = _mm256_fmadd_ps(sj, kj, sa8);
                }
                float sa = -sum256(sa8);

                /* S += S*w + v⊗k + sa⊗kka ; y_i = S·r  (fused AVX2) */
                __m256 vk  = _mm256_set1_ps(vi);
                __m256 sk  = _mm256_set1_ps(sa);
                __m256 acc = _mm256_setzero_ps();
                for (int j = 0; j < h_size; j += 8) {
                    __m256 s = _mm256_mul_ps(_mm256_loadu_ps(Si + j),
                                             _mm256_loadu_ps(w_val + j));
                    s = _mm256_fmadd_ps(vk, _mm256_loadu_ps(kt + off + j), s);
                    s = _mm256_fmadd_ps(sk, _mm256_loadu_ps(kka + j), s);
                    _mm256_storeu_ps(Si + j, s);
                    acc = _mm256_fmadd_ps(s, _mm256_loadu_ps(rt + off + j), acc);
                }
                yt[off + i] = sum256(acc);
            }

            /* snapshot this head's att_state after token t */
            if (snap)
                memcpy(snap + (size_t)t * snap_stride + (size_t)h * h_size * h_size,
                       S, (size_t)h_size * h_size * sizeof(float));
        }
    }
}

/* ── windowed forward: logits for ALL T tokens + per-token snapshots ─ */
void forward_window(
    const Weights &W, State &S,
    const int *tokens, int T,             /* T ≤ 8 */
    float *const logits_all,              /* [T, VOCAB] */
    float *const states_out)              /* [T * state_size] or NULL */
{
    int dim      = W.dim;
    int n_heads  = W.n_heads;
    int n_layers = W.n_layers;
    int vocab    = W.vocab;
    int ffn_dim  = W.ffn_dim;

    const size_t Ssz       = state_snapshot_size(n_layers, dim, n_heads);
    const size_t off_state = (size_t)n_layers * dim;
    const size_t off_ffn   = off_state + (size_t)n_layers * n_heads * h_size * h_size;

    int TC = T * dim;
    int TF = T * ffn_dim;

    /* same scratch layout as forward_seq */
    static AlignedBuffer buf_storage;
    size_t buf_size = 12 * TC + 2 * TF;
    buf_storage.resize(buf_size);
    float *const buf = buf_storage.data();

    float *const x      = buf;
    float *const r      = buf + 1 * TC;
    float *const wb     = buf + 2 * TC;
    float *const kb     = buf + 3 * TC;
    float *const vb_    = buf + 4 * TC;
    float *const ab     = buf + 5 * TC;
    float *const gb     = buf + 6 * TC;
    float *const kkb    = buf + 7 * TC;
    float *const tmp    = buf + 8 * TC;
    float *const tmp2   = buf + 9 * TC;
    float *const xv_s   = buf + 10 * TC;
    float *const ftmp   = buf + 11 * TC;
    float *const ftmp2  = buf + 11 * TC + TF;
    float *const v_first = buf + 11 * TC + 2 * TF;

    /* embedding: 16-bit → fp32 on the fly */
    for (int t = 0; t < T; t++) {
        float *__restrict const xt = x + t * dim;
        const uint16_t *__restrict const emb_t = W.emb + tokens[t] * dim;
        for (int i = 0; i < dim; i += 8)
            _mm256_storeu_ps(xt + i, load_16x8(emb_t + i));
    }

    for (int l = 0; l < n_layers; l++) {
        const Weights::Layer &L = W.layers[l];

        /* ── LN1 — snapshot att_x_prev[l] = ln1_out(t) ────────────── */
        for (int t = 0; t < T; t++)
            layer_norm(x + t * dim, tmp + t * dim, L.ln1_w, L.ln1_b, dim);
        if (states_out)
            for (int t = 0; t < T; t++)
                memcpy(states_out + t * Ssz + (size_t)l * dim,
                       tmp + t * dim, dim * sizeof(float));

        /* ── time-shift: xx[0] = x_prev - x[0], xx[t] = x[t-1] - x[t] ─ */
        {
            float *__restrict const dst0 = tmp2;
            const float *__restrict const now0 = tmp;
            const float *__restrict const prev_s = S.att_x_prev_at(l);
            for (int i = 0; i < dim; i += 8) {
                __m256 p = _mm256_loadu_ps(prev_s + i);
                __m256 n = _mm256_loadu_ps(now0 + i);
                _mm256_storeu_ps(dst0 + i, _mm256_sub_ps(p, n));
            }
        }
        for (int t = 1; t < T; t++) {
            const float *__restrict const prev_tok = tmp  + (t - 1) * dim;
            const float *__restrict const cur_tok  = tmp  + t * dim;
            float       *__restrict const dst      = tmp2 + t * dim;
            for (int i = 0; i < dim; i += 8) {
                __m256 p = _mm256_loadu_ps(prev_tok + i);
                __m256 c = _mm256_loadu_ps(cur_tok + i);
                _mm256_storeu_ps(dst + i, _mm256_sub_ps(p, c));
            }
        }
        {
            const float *__restrict const last = tmp + (T - 1) * dim;
            float *__restrict const prev_dst = S.att_x_prev_at(l);
            for (int i = 0; i < dim; i += 8)
                _mm256_storeu_ps(prev_dst + i, _mm256_loadu_ps(last + i));
        }

        /* ── shifted inputs ─────────────────────────────────────── */
        for (int t = 0; t < T; t++) {
            const float *const xt  = tmp  + t * dim;
            const float *const xxt = tmp2 + t * dim;
            time_shift(xt, xxt, L.x_r, r   + t * dim, dim);
            time_shift(xt, xxt, L.x_w, wb  + t * dim, dim);
            time_shift(xt, xxt, L.x_k, kb  + t * dim, dim);
            time_shift(xt, xxt, L.x_v, vb_ + t * dim, dim);
            time_shift(xt, xxt, L.x_a, ab  + t * dim, dim);
            time_shift(xt, xxt, L.x_g, gb  + t * dim, dim);
        }

        /* ── pre-compute v0 gate (before V-projection overwrites vb_) ── */
        if (l > 0) {
            int v1_hidden = L.v1_hidden;
            gemv_f16_batchT(vb_, L.v1, tmp, T, dim, v1_hidden);
            gemv_f16_batchT(tmp, L.v2, xv_s, T, v1_hidden, dim);
        }

        /* ── batched projections (16-bit weights streamed once) ────
         * batched-GEMV writes output directly (unlike the GEMM path's
         * staged C buffer), so the three in-place projections are
         * staged through tmp2 to avoid input/output aliasing. */
        gemv_f16_batchT(r,   L.R, tmp2, T, dim, dim); memcpy(r,   tmp2, TC * sizeof(float));
        gemv_f16_batchT(kb,  L.K, tmp2, T, dim, dim); memcpy(kb,  tmp2, TC * sizeof(float));
        gemv_f16_batchT(vb_, L.V, tmp2, T, dim, dim); memcpy(vb_, tmp2, TC * sizeof(float));

        /* w */
        {
            int w1_hidden = L.w1_hidden;
            gemv_f16_batchT(wb, L.w1, tmp, T, dim, w1_hidden);
            for (int i = 0; i < T * w1_hidden; i++) tmp[i] = tanhf(tmp[i]);
            gemv_f16_batchT(tmp, L.w2, wb, T, w1_hidden, dim);
        }

        /* a */
        {
            int a1_hidden = L.a1_hidden;
            gemv_f16_batchT(ab, L.a1, tmp, T, dim, a1_hidden);
            gemv_f16_batchT(tmp, L.a2, tmp2, T, a1_hidden, dim);
            for (int t = 0; t < T; t++)
                for (int i = 0; i < dim; i++)
                    ab[t * dim + i] = sigmoid(L.a0[i] + tmp2[t * dim + i]);
        }

        /* g */
        {
            int g1_hidden = L.g1_hidden;
            gemv_f16_batchT(gb, L.g1, tmp, T, dim, g1_hidden);
            for (int i = 0; i < T * g1_hidden; i++) tmp[i] = sigmoid(tmp[i]);
            gemv_f16_batchT(tmp, L.g2, gb, T, g1_hidden, dim);
        }

        /* ── K normalize ────────────────────────────────────────── */
        k_normalize(kb, ab, kkb, L.k_k, L.k_a, T, dim, n_heads, h_size);

        /* ── v0 adjustment ──────────────────────────────────────── */
        if (l == 0) {
            memcpy(v_first, vb_, TC * sizeof(float));
        } else {
            for (int t = 0; t < T; t++) {
                float *const vt = vb_ + t * dim;
                const float *const vf = v_first + t * dim;
                const float *const res = xv_s + t * dim;
                for (int i = 0; i < dim; i++)
                    vt[i] += (vf[i] - vt[i]) * sigmoid(L.v0[i] + res[i]);
            }
        }

        /* ── WKV (window + per-token att_state snapshots) ────────── */
        wkv_window(r, wb, kb, vb_, kkb, ab, L.w0,
                   S.att_state_at(l), tmp, T, dim, n_heads, h_size,
                   states_out ? states_out + off_state
                               + (size_t)l * n_heads * h_size * h_size : nullptr,
                   Ssz);

        /* ── group norm + bonus + output — ALL tokens ────────────── */
        for (int t = 0; t < T; t++)
            group_norm(tmp + t * dim, tmp2 + t * dim,
                       L.ln_x_w, L.ln_x_b, n_heads, h_size);

        for (int t = 0; t < T; t++) {
            float       *const yt = tmp2 + t * dim;
            const float *const rt = r    + t * dim;
            const float *const kt = kb   + t * dim;
            const float *const vt = vb_  + t * dim;
            for (int h = 0; h < n_heads; h++) {
                float bonus = 0;
                for (int j = 0; j < h_size; j++) {
                    int idx = h * h_size + j;
                    bonus += rt[idx] * kt[idx] * L.r_k[idx];
                }
                for (int j = 0; j < h_size; j++)
                    yt[h * h_size + j] += bonus * vt[h * h_size + j];
            }
        }

        for (int t = 0; t < T; t++)
            vec_mul(tmp2 + t * dim, gb + t * dim, dim);
        gemv_f16_batchT(tmp2, L.O, tmp, T, dim, dim);
        for (int t = 0; t < T; t++)
            vec_add(x + t * dim, tmp + t * dim, dim);

        /* ── LN2 — snapshot ffn_x_prev[l] = ln2_out(t) ───────────── */
        for (int t = 0; t < T; t++)
            layer_norm(x + t * dim, tmp + t * dim, L.ln2_w, L.ln2_b, dim);
        if (states_out)
            for (int t = 0; t < T; t++)
                memcpy(states_out + t * Ssz + off_ffn + (size_t)l * dim,
                       tmp + t * dim, dim * sizeof(float));

        /* FFN time-shift: sequential per-token (d[t] = prev - x[t]) */
        for (int t = 0; t < T; t++) {
            float *const xt = tmp  + t * dim;
            float *const d  = tmp2 + t * dim;
            float *const fxp = S.ffn_x_prev_at(l);
            for (int i = 0; i < dim; i++) {
                d[i] = fxp[i] - xt[i];
                fxp[i] = xt[i];
            }
            time_shift(xt, d, L.ffn_x_k, kb + t * dim, dim);
        }

        gemv_f16_batchT(kb, L.ffn_K, ftmp, T, dim, ffn_dim);
        relu_sq(ftmp2, ftmp, TF);
        /* V projection exploits ReLU² sparsity (≈88% zeros):
         * batched sparse kernel touches only the union of non-zero rows */
        gemv_f16_sparse_batchT(ftmp2, L.ffn_V, tmp, T, ffn_dim, dim);
        for (int t = 0; t < T; t++)
            vec_add(x + t * dim, tmp + t * dim, dim);
    }

    /* ── final LN + head for ALL tokens ───────────────────────── */
    for (int t = 0; t < T; t++)
        layer_norm(x + t * dim, tmp + t * dim, W.ln_out_w, W.ln_out_b, dim);
    gemv_f16_batchT(tmp, W.head, logits_all, T, dim, vocab);
}

/* ── speculative prefill: any T, processed in windows of ≤ 8 ────── */
void forward_spec(
    const Weights &W, State &S,
    const int *tokens, int T,
    float *const logits_all, float *const states_out)
{
    const int    CH  = 8;
    const size_t Ssz = state_snapshot_size(W.n_layers, W.dim, W.n_heads);
    for (int done = 0; done < T; ) {
        int n = (T - done < CH) ? (T - done) : CH;
        forward_window(W, S, tokens + done, n,
                       logits_all + (size_t)done * W.vocab,
                       states_out ? states_out + (size_t)done * Ssz : nullptr);
        done += n;
    }
}

/* ════════════════════════════════════════════════════════════════════
 * C API  (suppressed when embedded by src/rwkv_api.cpp, which owns
 * the public entry points and dispatches to Fp16::/Bf16:: instances)
 * ════════════════════════════════════════════════════════════════════ */

#ifndef RWKV_ENGINE_EMBEDDED

extern "C" {

RWKV_API RWKVWeights *rwkv_weights_load(const char *path)
{
    Weights *w = new Weights();
    if (!w->load(path)) { delete w; return nullptr; }
    return (RWKVWeights *)w;
}

RWKV_API void rwkv_weights_free(RWKVWeights *w)
{
    delete (Weights *)w;
}

/* ── Dimension accessors ─────────────────────────────────────────── */
RWKV_API int rwkv_get_dim(RWKVWeights *w)
    { return ((Weights *)w)->dim; }
RWKV_API int rwkv_get_n_heads(RWKVWeights *w)
    { return ((Weights *)w)->n_heads; }
RWKV_API int rwkv_get_h_size(RWKVWeights *w)
    { return 64; }
RWKV_API int rwkv_get_n_layers(RWKVWeights *w)
    { return ((Weights *)w)->n_layers; }
RWKV_API int rwkv_get_vocab(RWKVWeights *w)
    { return ((Weights *)w)->vocab; }
RWKV_API int rwkv_get_ffn_dim(RWKVWeights *w)
    { return ((Weights *)w)->ffn_dim; }

/* ── State ─────────────────────────────────────────────────────── */
RWKV_API RWKVState *rwkv_state_new(RWKVWeights *w)
{
    Weights *W = (Weights *)w;
    return (RWKVState *)(new State(W->dim, W->n_heads, 64, W->n_layers));
}

RWKV_API void rwkv_state_free(RWKVState *s)
{
    delete (State *)s;
}

RWKV_API void rwkv_state_reset(RWKVState *s)
{
    if (s) ((State *)s)->reset();
}

RWKV_API void rwkv_decode(
    RWKVWeights *w, RWKVState *s,
    int token, float *logits)
{
    forward_one(*(Weights *)w, *(State *)s, token, logits);
}

RWKV_API void rwkv_prefill(
    RWKVWeights *w, RWKVState *s,
    const int *tokens, int T,
    float *logits)
{
    forward_seq(*(Weights *)w, *(State *)s, tokens, T, logits);
}

/* ── Speculative decoding ─────────────────────────────────────── */
RWKV_API int rwkv_state_size(RWKVWeights *w)
{
    Weights *W = (Weights *)w;
    return (int)state_snapshot_size(W->n_layers, W->dim, W->n_heads);
}

RWKV_API void rwkv_state_save(RWKVState *s, float *dst)
{
    if (s) state_save(*(State *)s, dst);
}

RWKV_API void rwkv_state_restore(RWKVState *s, const float *src)
{
    if (s) state_restore(*(State *)s, src);
}

RWKV_API void rwkv_spec_prefill(
    RWKVWeights *w, RWKVState *s,
    const int *tokens, int T,
    float *logits, float *states)
{
    forward_spec(*(Weights *)w, *(State *)s, tokens, T,
                 logits, states);
}

RWKV_API void rwkv_debug_embed(RWKVWeights *w, int token, float *out)
{
    Weights *W = (Weights *)w;
    for (int i = 0; i < W->dim; i += 8)
        _mm256_storeu_ps(out + i, load_16x8(W->emb + token * W->dim + i));
}

} // extern "C"

#endif /* RWKV_ENGINE_EMBEDDED */
