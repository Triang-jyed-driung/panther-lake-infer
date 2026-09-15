// llm_sampler.h
//
// 基于 prefix_sum_v2 的 CPU LLM 采样器（温度 + top-p，nucleus）：
//
//     logits ──温度+softmax──> probs ──top-p 修正（禁止排序，四分法）──> 修正后 probs
//        ──build_weights（实例化 int 权重 wi）──> wi ──build_local_prefixes──> prefix_sum
//        ──categorical_sample(v2)──> token id
//
// 约束（遵循设计约定）：
//   * 禁止排序：top-p 采用「4 层指数直方图定位 + 边界补偿」（阈值由位模式直接拼出）。
//   * 禁止 double：全程 float / 整数（RNG 用 uint64 splitmix64）。
//   * 全 AVX2 SIMD；float<->uint32 用 std::bit_cast（不用 memcpy）。
//   * 全部静态数组（无 vector/堆分配），编译期固定 MAX_N。
//   * n >= 4096 走 4 线程路径：常驻 pthread 线程池（主线程 + 3 worker，绑定 0-3 号大核，
//     大小核架构勿用 8 线程），无 OpenMP fork-join；线程创建失败直接终止（无法降级）。
//   * n < 4096 走单线程 AVX2 路径。
//
// 注意：LLMSampler 对象较大（~12MB @ MAX_N=1<<20：probs_/weights_/prefix_sum_ 三个 4MB 数组），
//       请放堆/静态，勿放栈。线程池的 3 个 worker 在两次采样之间自旋等待（约占用 3 核）。
#pragma once

#include <cstdint>
#include <cstddef>
#include <bit>
#include <cmath>
#include <algorithm>
#include <limits>
#include <tuple>
#include <immintrin.h>
#include <pthread.h>
#include <cstdlib>
#ifdef __linux__
#include <sched.h>
#endif

#include "prefix_sum_v2.cpp"

namespace llm_sampling {

static constexpr size_t MAX_N = 1u << 20;   // 最大词表/序列长度

// ===========================================================================
// top-p 4 层直方图位划分（真实 SIMD 实测最优：(10,8,9,4)）
// ===========================================================================
static constexpr int SPLIT_W1 = 10, SPLIT_W2 = 8, SPLIT_W3 = 9, SPLIT_W4 = 4;
static constexpr int HIST1_BUCKETS = 1 << SPLIT_W1;   // 1024
static constexpr int HIST2_BUCKETS = 1 << SPLIT_W2;   // 256
static constexpr int HIST3_BUCKETS = 1 << SPLIT_W3;   // 512
static constexpr int HIST4_BUCKETS = 1 << SPLIT_W4;   // 16
static constexpr int SHIFT1 = 31 - SPLIT_W1;          // 21：级1 取位 30..21
static constexpr int SHIFT2 = 31 - SPLIT_W1 - SPLIT_W2; // 13：级2 取位 20..13
static constexpr int SHIFT3 = SPLIT_W4;               // 4：级3 取位 12..4
static constexpr uint32_t MASK1 = HIST1_BUCKETS - 1u;
static constexpr uint32_t MASK2 = HIST2_BUCKETS - 1u;
static constexpr uint32_t MASK3 = HIST3_BUCKETS - 1u;
static constexpr uint32_t MASK4 = HIST4_BUCKETS - 1u;

// ===========================================================================
// 数值工具（全程 float / 整数）
// ===========================================================================

static inline uint32_t f2u(float f) { return *(uint32_t*)&f; }
static inline float u2f(uint32_t u) { return *(float*)&u; }

// NaN → 0，±Inf → ±FLT_MAX
static inline float sf(float x) {
    float y = std::isnan(x) ? 0.0f : x;
    return (std::isinf(y) ? std::copysignf(std::numeric_limits<float>::max(), y) : y);
}

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static inline float hmax256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_max_ps(lo, hi);
    s = _mm_max_ps(s, _mm_shuffle_ps(s, s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_max_ps(s, _mm_shuffle_ps(s, s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtss_f32(s);
}

// exp2(x) AVX2：floor 分整 + 5 阶多项式 + 位构造 2^i（相对误差 ~1e-6）
static inline __m256 exp2v(__m256 x) {
    const __m256 c0 = _mm256_set1_ps(1.0f),        c1 = _mm256_set1_ps(0.69314718f);
    const __m256 c2 = _mm256_set1_ps(0.24022651f), c3 = _mm256_set1_ps(0.05550411f);
    const __m256 c4 = _mm256_set1_ps(0.00961813f), c5 = _mm256_set1_ps(0.00133336f);
    const __m256 c127 = _mm256_set1_ps(127.0f);
    x = _mm256_max_ps(x, _mm256_set1_ps(-150.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(127.0f));
    __m256 t = _mm256_floor_ps(x);
    __m256 f = _mm256_sub_ps(x, t);
    __m256 p = _mm256_fmadd_ps(c5, f, c4);
    p = _mm256_fmadd_ps(p, f, c3);
    p = _mm256_fmadd_ps(p, f, c2);
    p = _mm256_fmadd_ps(p, f, c1);
    p = _mm256_fmadd_ps(p, f, c0);
    __m256i ti = _mm256_cvtps_epi32(_mm256_add_ps(t, c127));
    ti = _mm256_max_epi32(ti, _mm256_set1_epi32(1));
    __m256 sc = _mm256_castsi256_ps(_mm256_slli_epi32(ti, 23));
    return _mm256_mul_ps(p, sc);
}

// splitmix64：纯整数 RNG
static inline uint64_t next_rand(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// ===========================================================================
// SIMD 直方图辅助函数（消除 execute_body 中 3 处重复的提取 + 分桶逻辑）
// ===========================================================================

// 从 __m256 提取 8 个 float（纯位操作，无精度损失）
__attribute__((always_inline)) static inline void extract8(__m256 p, float out[8]) {
    __m128 lo = _mm256_castps256_ps128(p), hi = _mm256_extractf128_ps(p, 1);
    out[0] = _mm_cvtss_f32(lo);
    out[1] = _mm_cvtss_f32(_mm_permute_ps(lo, 0x55));
    out[2] = _mm_cvtss_f32(_mm_permute_ps(lo, 0xAA));
    out[3] = _mm_cvtss_f32(_mm_permute_ps(lo, 0xFF));
    out[4] = _mm_cvtss_f32(hi);
    out[5] = _mm_cvtss_f32(_mm_permute_ps(hi, 0x55));
    out[6] = _mm_cvtss_f32(_mm_permute_ps(hi, 0xAA));
    out[7] = _mm_cvtss_f32(_mm_permute_ps(hi, 0xFF));
}

// 把 8 个 float 全部分桶到 hist（无条件，用于级1）
__attribute__((always_inline)) static inline void bucket8_all(const float fs[8], float* hist,
                                int shift, uint32_t mask) {
    for (int i = 0; i < 8; i++)
        hist[(f2u(fs[i]) >> shift) & mask] += fs[i];
}

// 把 mask 指定的 float 分桶到 hist（用于级2）
__attribute__((always_inline)) static inline void bucket8_masked(const float fs[8], int mask, float* hist,
                                   int shift, uint32_t bit_mask) {
    for (int i = 0; i < 8; i++)
        if (mask & (1 << i))
            hist[(f2u(fs[i]) >> shift) & bit_mask] += fs[i];
}

// 把 mask 指定的 float 同时分桶到 h3（按 SHIFT3）和 h4（按 MASK4）并计数（用于级3+4）
__attribute__((always_inline)) static inline void bucket8_dual(const float fs[8], int mask,
                                 float* h3, float* h4, int* c4) {
    for (int i = 0; i < 8; i++) {
        if (mask & (1 << i)) {
            uint32_t uu = f2u(fs[i]);
            h3[(uu >> SHIFT3) & MASK3] += fs[i];
            h4[uu & MASK4] += fs[i];
            ++c4[uu & MASK4];
        }
    }
}

// 标量版：把 probs[start..end) 分桶到 hist（用于尾部处理）
static inline void bucket_scalar(const float* probs, size_t start, size_t end,
                                  float* hist, int shift, uint32_t mask) {
    for (size_t i = start; i < end; i++)
        hist[(f2u(probs[i]) >> shift) & mask] += probs[i];
}

// 标量版：带条件分桶（级2+）
static inline void bucket_scalar_cond(const float* probs, size_t start, size_t end,
                                       uint32_t cond_shift, uint32_t cond_val,
                                       float* hist, int shift, uint32_t mask) {
    for (size_t i = start; i < end; i++) {
        uint32_t u = f2u(probs[i]);
        if ((u >> cond_shift) == cond_val)
            hist[(u >> shift) & mask] += probs[i];
    }
}

// 标量版：带双条件分桶到 h4+c4，可选写入 h3（级3+4 共用）
static inline void bucket_scalar_dual(const float* probs, size_t start, size_t end,
                                       uint32_t c1_shift, uint32_t c1_val,
                                       uint32_t c2_shift, uint32_t c2_mask, uint32_t c2_val,
                                       float* h3, float* h4, int* c4) {
    for (size_t i = start; i < end; i++) {
        uint32_t u = f2u(probs[i]);
        if ((u >> c1_shift) == c1_val && ((u >> c2_shift) & c2_mask) == c2_val) {
            if (h3) h3[(u >> SHIFT3) & MASK3] += probs[i];
            h4[u & MASK4] += probs[i];
            ++c4[u & MASK4];
        }
    }
}

// ===========================================================================
// softmax + 温度缩放（单线程，4 块并行）
// ===========================================================================

static inline void softmax_temperature_serial(const float* logits, size_t n,
                                               float temperature, float* probs) {
    if (n == 0) return;
    const float inv = 1.4426950408889634f / temperature;
    const size_t chunk = (n + 3) / 4;

    float max_arr[4];
    for (size_t t = 0; t < 4; ++t) {
        const size_t start = t * chunk, end = std::min(start + chunk, n);
        __m256 mx = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        __m256 invv = _mm256_set1_ps(inv);
        size_t i = start;
        for (; i + 8 <= end; i += 8)
            mx = _mm256_max_ps(mx, _mm256_mul_ps(_mm256_loadu_ps(logits + i), invv));
        float tm = -std::numeric_limits<float>::infinity();
        for (; i < end; ++i) tm = std::max(tm, sf(logits[i] * inv));
        max_arr[t] = std::max(hmax256(mx), tm);
    }
    float maxu = max_arr[0];
    for (int t = 1; t < 4; ++t) maxu = std::max(maxu, max_arr[t]);

    float den_arr[4];
    __m256 invv2 = _mm256_set1_ps(inv);
    __m256 gmv = _mm256_set1_ps(maxu);
    for (size_t t = 0; t < 4; ++t) {
        const size_t start = t * chunk, end = std::min(start + chunk, n);
        __m256 ds = _mm256_setzero_ps();
        size_t i = start;
        for (; i + 8 <= end; i += 8) {
            __m256 e = exp2v(_mm256_sub_ps(_mm256_mul_ps(_mm256_loadu_ps(logits + i), invv2), gmv));
            _mm256_storeu_ps(probs + i, e);
            ds = _mm256_add_ps(ds, e);
        }
        float td = 0;
        for (; i < end; ++i) { float e = std::exp2(sf(logits[i] * inv) - maxu); probs[i] = e; td += e; }
        den_arr[t] = hsum256(ds) + td;
    }
    float denom = den_arr[0];
    for (int t = 1; t < 4; ++t) denom += den_arr[t];
    const float invd = 1.0f / denom;
    __m256 invdv = _mm256_set1_ps(invd);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(probs + i, _mm256_mul_ps(_mm256_loadu_ps(probs + i), invdv));
    for (; i < n; ++i) probs[i] *= invd;
}

// ===========================================================================
// top-p 分析（单线程版）：4 层直方图定位 + 边界补偿
// ===========================================================================
struct TopPInfo { float th, comp, total; };

// 4 块分块版：与并行路径 execute_body 的直方图构建/合并顺序逐位一致
static inline TopPInfo top_p_analyze_serial(const float* probs, size_t n, float top_p) {
    static thread_local float h1[4][HIST1_BUCKETS], h2[4][HIST2_BUCKETS];
    static thread_local float h3[4][HIST3_BUCKETS], h4[4][HIST4_BUCKETS];
    static thread_local int   c4[4][HIST4_BUCKETS];
    TopPInfo r{0.0f, 1.0f, 1.0f};
    const size_t chunk = (n + 3) / 4;

    // ---- 级1：直方图（位 30-21），4 块构建 + 固定合并序 ----
    for (int t = 0; t < 4; ++t) {
        const size_t start = t * chunk, end = std::min(start + chunk, n);
        float* hh = h1[t];
        for (int b = 0; b < HIST1_BUCKETS; ++b) hh[b] = 0.0f;
        bucket_scalar(probs, start, end, hh, SHIFT1, MASK1);
    }
    float suffix = 0.0f, suffix_above = 0.0f;
    int be = HIST1_BUCKETS - 1;
    for (int b = HIST1_BUCKETS - 1; b >= 0; --b) {
        float s = 0.0f;
        for (int t = 0; t < 4; ++t) s += h1[t][b];
        suffix += s;
        if (suffix > top_p) { be = b; suffix_above = suffix - s; break; }
    }
    if (be == HIST1_BUCKETS - 1 && suffix <= top_p) return r;

    // ---- 级2：子直方图（位 20-13）----
    for (int t = 0; t < 4; ++t) {
        const size_t start = t * chunk, end = std::min(start + chunk, n);
        float* hh = h2[t];
        for (int b = 0; b < HIST2_BUCKETS; ++b) hh[b] = 0.0f;
        bucket_scalar_cond(probs, start, end, SHIFT1, (uint32_t)be, hh, SHIFT2, MASK2);
    }
    const float tgt1 = top_p - suffix_above;
    suffix = 0.0f;
    int sbin = HIST2_BUCKETS - 1;
    float sbin_sum = 0.0f;
    for (int b = HIST2_BUCKETS - 1; b >= 0; --b) {
        float s = 0.0f;
        for (int t = 0; t < 4; ++t) s += h2[t][b];
        suffix += s;
        if (suffix > tgt1) { sbin = b; sbin_sum = s; break; }
    }
    const float s_above2 = suffix - sbin_sum;
    const float tgt = top_p - suffix_above - s_above2;

    // ---- 级3：直方图（位 12-4），双条件 be+sbin ----
    for (int t = 0; t < 4; ++t) {
        const size_t start = t * chunk, end = std::min(start + chunk, n);
        float* hh = h3[t];
        for (int b = 0; b < HIST3_BUCKETS; ++b) hh[b] = 0.0f;
        for (size_t i = start; i < end; ++i) {
            uint32_t u = f2u(probs[i]);
            if ((u >> SHIFT1) == (uint32_t)be && ((u >> SHIFT2) & MASK2) == (uint32_t)sbin)
                hh[(u >> SHIFT3) & MASK3] += probs[i];
        }
    }
    suffix = 0.0f;
    int tbin3 = 0;
    float sbin3_sum = 0.0f;
    for (int b = HIST3_BUCKETS - 1; b >= 0; --b) {
        float s = 0.0f;
        for (int t = 0; t < 4; ++t) s += h3[t][b];
        suffix += s;
        if (suffix >= tgt) { tbin3 = b; sbin3_sum = s; break; }
    }
    const float s_above3 = suffix - sbin3_sum;
    const float tgt3 = tgt - s_above3;

    // ---- 级4：直方图 + 计数（位 3-0）----
    for (int t = 0; t < 4; ++t) {
        const size_t start = t * chunk, end = std::min(start + chunk, n);
        float* hh = h4[t]; int* cc = c4[t];
        for (int b = 0; b < HIST4_BUCKETS; ++b) { hh[b] = 0.0f; cc[b] = 0; }
        bucket_scalar_dual(probs, start, end,
                           SHIFT1, (uint32_t)be, SHIFT2, MASK2, (uint32_t)sbin,
                           nullptr, hh, cc);  // h3 已构建，这里只用 h4+c4
    }
    suffix = 0.0f;
    int tbin4 = 0;
    float sbin4_sum = 0.0f;
    for (int b = HIST4_BUCKETS - 1; b >= 0; --b) {
        float s = 0.0f;
        for (int t = 0; t < 4; ++t) s += h4[t][b];
        suffix += s;
        if (suffix >= tgt3) { tbin4 = b; sbin4_sum = s; break; }
    }
    const float s_above4 = suffix - sbin4_sum;
    int eqk = 0;
    for (int t = 0; t < 4; ++t) eqk += c4[t][tbin4];

    // ---- 32 位位模式拼出 th，边界补偿 ----
    const uint32_t th_bits = ((uint32_t)be << SHIFT1) | ((uint32_t)sbin << SHIFT2)
                           | ((uint32_t)tbin3 << SHIFT3) | (uint32_t)tbin4;
    r.th = u2f(th_bits);
    const float gtp = s_above3 + s_above4;
    if (r.th > 0.0f && eqk > 0) {
        r.comp = std::min(sf((tgt - gtp) / (r.th * (float)eqk)), 1.0f);
        r.comp = std::max(r.comp, 0.0f);
        r.total = suffix_above + s_above2 + gtp + r.th * (float)eqk * r.comp;
    } else {
        r.th = 0.0f; r.comp = 1.0f; r.total = 1.0f;
    }
    return r;
}

// ===========================================================================
// 单行采样管线（public）：logits → 前缀和，返回总权重
// ===========================================================================
static inline int build_row_weights(const float* logits, size_t n,
                                    float temperature, float top_p,
                                    float* probs, int* weights,
                                    int* prefix, int* block_sums) {
    softmax_temperature_serial(logits, n, temperature, probs);
    TopPInfo info{0.0f, 1.0f, 1.0f};
    if (top_p < 1.0f) info = top_p_analyze_serial(probs, n, top_p);
    const float factor = TARGET_TOTAL_F / (info.total > 0.0f ? info.total : 1.0f);
    categorical_build_weights(probs, n, weights, factor, info.th, info.comp, 0, 1);
    categorical_build_local_prefixes(weights, n, prefix, block_sums, 0, 1);
    categorical_add_local_offsets(prefix, n, block_sums, 0, 1);
    return categorical_total_weight(prefix, n);
}

// ===========================================================================
// 投机解码验证（speculative sampling，int 权重）
// ===========================================================================
static inline std::tuple<int, int> speculative_verify(
    const int* weights_P, const int* weights_Q, const int* spec_ids,
    size_t m, size_t n, int* residual, int* prefix_sum, int* block_sums,
    uint64_t& rng) {
    for (size_t a = 0; a < m; ++a) {
        const int* P = weights_P + a * n;
        const int* Q = weights_Q + a * n;
        const int id = spec_ids[a];
        bool reject = (id < 0 || (size_t)id >= n);
        if (!reject) {
            const int u = P[id], v = Q[id];
            if (u > v) reject = (int)(next_rand(rng) % (uint64_t)u) >= v;
        }
        if (!reject) continue;

        // 残差分布 max(Q-P, 0)
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256i q = _mm256_loadu_si256((const __m256i*)(Q + i));
            __m256i p = _mm256_loadu_si256((const __m256i*)(P + i));
            _mm256_storeu_si256((__m256i*)(residual + i),
                _mm256_max_epi32(_mm256_sub_epi32(q, p), _mm256_setzero_si256()));
        }
        for (; i < n; ++i) { int d = Q[i] - P[i]; residual[i] = d > 0 ? d : 0; }

        categorical_build_local_prefixes(residual, n, prefix_sum, block_sums, 0, 1);
        categorical_add_local_offsets(prefix_sum, n, block_sums, 0, 1);
        int total = categorical_total_weight(prefix_sum, n);
        if (total <= 0) continue;
        return {(int)a, (int)categorical_sample(prefix_sum, n, total, next_rand(rng))};
    }
    return {m, -1};
}

// ===========================================================================
// 采样器（全部静态数组；常驻线程池）
// ===========================================================================
class LLMSampler {
public:
    explicit LLMSampler(size_t vocab_size) {
        (void)vocab_size;
        pthread_mutex_init(&m_, nullptr);
        pthread_cond_init(&c_, nullptr);
#ifdef __linux__
        cpu_set_t cs0; CPU_ZERO(&cs0); CPU_SET(0, &cs0);
        sched_setaffinity(0, sizeof(cs0), &cs0);
#endif
        for (int w = 1; w < NT; ++w) {
            worker_arg_[w - 1].self = this;
            worker_arg_[w - 1].tid = w;
            if (pthread_create(&workers_[w - 1], nullptr, worker_main, &worker_arg_[w - 1]) != 0) abort();
#ifdef __linux__
            cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(w, &cs);
            pthread_setaffinity_np(workers_[w - 1], sizeof(cs), &cs);
#endif
        }
    }

    ~LLMSampler() {
        for (int w = 1; w < NT; ++w)
            while (__atomic_load_n(&done_[w], __ATOMIC_ACQUIRE) != publish_) _mm_pause();
        pthread_mutex_lock(&m_);
        publish_ = STOP_EPOCH;
        pthread_cond_broadcast(&c_);
        pthread_mutex_unlock(&m_);
        for (int w = 0; w < NT - 1; ++w) pthread_join(workers_[w], nullptr);
        pthread_mutex_destroy(&m_);
        pthread_cond_destroy(&c_);
    }

    // 完整 pipeline：logits -> token id
    size_t sample(const float* logits, size_t n,
                  float temperature, float top_p, uint64_t& rng_state) {
        if (n == 0 || n > MAX_N) return 0;
        if (top_p <= 0.0f) {
            size_t best = 0;
            float bv = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < n; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
            return best;
        }
        if (n < PAR_THRESHOLD) return sample_serial(logits, n, temperature, top_p, rng_state);

        // 大 n：常驻线程池
        for (int w = 1; w < NT; ++w)
            while (__atomic_load_n(&done_[w], __ATOMIC_ACQUIRE) != publish_) _mm_pause();
        task_.logits = logits; task_.n = n; task_.temperature = temperature;
        task_.top_p = top_p; task_.rng = &rng_state;
        pthread_mutex_lock(&m_);
        publish_ = publish_ + 1;
        pthread_cond_broadcast(&c_);
        pthread_mutex_unlock(&m_);
        execute_body(0, publish_);
        return task_.token;
    }

public:
    static constexpr size_t PAR_THRESHOLD = 4096;
    static constexpr int NT = 4;
    static constexpr uint64_t STOP_EPOCH = ~0ULL;

    struct alignas(64) SpinSlot { uint64_t v[8]; };
    struct TaskParams {
        const float* logits; size_t n; float temperature, top_p;
        uint64_t* rng; size_t token;
    };
    struct WorkerArg { LLMSampler* self; int tid; };

    static void* worker_main(void* arg) {
        WorkerArg* a = static_cast<WorkerArg*>(arg);
        LLMSampler* self = a->self;
        const int tid = a->tid;
        uint64_t last = 0;
        pthread_mutex_lock(&self->m_);
        for (;;) {
            while (self->publish_ == last) pthread_cond_wait(&self->c_, &self->m_);
            last = self->publish_;
            if (last == STOP_EPOCH) { pthread_mutex_unlock(&self->m_); break; }
            pthread_mutex_unlock(&self->m_);
            self->execute_body(tid, last);
            __atomic_store_n(&self->done_[tid], last, __ATOMIC_RELEASE);
            pthread_mutex_lock(&self->m_);
        }
        return nullptr;
    }

    // 小 n：单线程 AVX2
    size_t sample_serial(const float* logits, size_t n,
                         float temperature, float top_p, uint64_t& rng_state) {
        softmax_temperature_serial(logits, n, temperature, probs_);
        TopPInfo info{0.0f, 1.0f, 1.0f};
        if (top_p < 1.0f) info = top_p_analyze_serial(probs_, n, top_p);
        const float factor = TARGET_TOTAL_F / (info.total > 0.0f ? info.total : 1.0f);
        categorical_build_weights(probs_, n, weights_, factor, info.th, info.comp, 0, 1);
        categorical_build_local_prefixes(weights_, n, prefix_sum_, block_sums_, 0, 1);
        categorical_add_local_offsets(prefix_sum_, n, block_sums_, 0, 1);
        const int total = categorical_total_weight(prefix_sum_, n);
        return categorical_sample(prefix_sum_, n, total, next_rand(rng_state));
    }

    // 大 n：每线程分块执行
    void execute_body(int tid, uint64_t ep) {
        const TaskParams& t = task_;
        const float* logits = t.logits;
        const size_t n = t.n;
        const float inv = 1.4426950408889634f / t.temperature;
        const float top_p = t.top_p;
        const size_t chunk = (n + NT - 1) / NT;
        const size_t start = (size_t)tid * chunk;
        const size_t end = std::min(start + chunk, n);

        float gm = 0, gd = 0, suffix_above = 0;
        int be = HIST1_BUCKETS - 1, sbin = HIST2_BUCKETS - 1;
        bool need = false;

        // ============ pass1: 缩放 + max ============
        {
            __m256 mx = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
            __m256 invv = _mm256_set1_ps(inv);
            size_t i = start;
            for (; i + 8 <= end; i += 8)
                mx = _mm256_max_ps(mx, _mm256_mul_ps(_mm256_loadu_ps(logits + i), invv));
            float tm = -std::numeric_limits<float>::infinity();
            for (; i < end; ++i) tm = std::max(tm, sf(logits[i] * inv));
            max_arr_[tid] = std::max(hmax256(mx), tm);
        }
        spin_barr(0, tid, ep);
        gm = max_arr_[0];
        for (int t = 1; t < NT; ++t) gm = std::max(gm, max_arr_[t]);

        // ============ pass2: 缩放 + exp2 + denom ============
        {
            __m256 gmv = _mm256_set1_ps(gm), ds = _mm256_setzero_ps(), invv = _mm256_set1_ps(inv);
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                __m256 e = exp2v(_mm256_sub_ps(_mm256_mul_ps(_mm256_loadu_ps(logits + i), invv), gmv));
                _mm256_storeu_ps(probs_ + i, e);
                ds = _mm256_add_ps(ds, e);
            }
            float td = 0;
            for (; i < end; ++i) { float e = std::exp2(sf(logits[i] * inv) - gm); probs_[i] = e; td += e; }
            den_arr_[tid] = hsum256(ds) + td;
        }
        spin_barr(1, tid, ep);
        gd = den_arr_[0];
        for (int t = 1; t < NT; ++t) gd += den_arr_[t];
        const float invd = 1.0f / gd;

        // ============ pass3: 归一化 + 级1 直方图 ============
        {
            __m256 invdv = _mm256_set1_ps(invd);
            float* h = hist_ + tid * HIST1_BUCKETS;
            for (int b = 0; b < HIST1_BUCKETS; ++b) h[b] = 0.0f;
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                __m256 p = _mm256_mul_ps(_mm256_loadu_ps(probs_ + i), invdv);
                _mm256_storeu_ps(probs_ + i, p);
                __m128 lo = _mm256_castps256_ps128(p), hi = _mm256_extractf128_ps(p, 1);
                float f0 = _mm_cvtss_f32(lo);
                float f1 = _mm_cvtss_f32(_mm_permute_ps(lo, 0x55));
                float f2 = _mm_cvtss_f32(_mm_permute_ps(lo, 0xAA));
                float f3 = _mm_cvtss_f32(_mm_permute_ps(lo, 0xFF));
                float f4 = _mm_cvtss_f32(hi);
                float f5 = _mm_cvtss_f32(_mm_permute_ps(hi, 0x55));
                float f6 = _mm_cvtss_f32(_mm_permute_ps(hi, 0xAA));
                float f7 = _mm_cvtss_f32(_mm_permute_ps(hi, 0xFF));
                h[(f2u(f0) >> SHIFT1) & MASK1] += f0;
                h[(f2u(f1) >> SHIFT1) & MASK1] += f1;
                h[(f2u(f2) >> SHIFT1) & MASK1] += f2;
                h[(f2u(f3) >> SHIFT1) & MASK1] += f3;
                h[(f2u(f4) >> SHIFT1) & MASK1] += f4;
                h[(f2u(f5) >> SHIFT1) & MASK1] += f5;
                h[(f2u(f6) >> SHIFT1) & MASK1] += f6;
                h[(f2u(f7) >> SHIFT1) & MASK1] += f7;
            }
            for (; i < end; ++i) {
                float p = probs_[i] * invd;
                probs_[i] = p;
                h[(f2u(p) >> SHIFT1) & MASK1] += p;
            }
        }
        spin_barr(2, tid, ep);

        // ============ 级1 分析（每线程冗余）============
        {
            float suffix = 0.0f;
            for (int b = HIST1_BUCKETS - 1; b >= 0; --b) {
                float s = 0.0f;
                for (int t = 0; t < NT; ++t) s += hist_[t * HIST1_BUCKETS + b];
                suffix += s;
                if (suffix > top_p) { need = true; be = b; suffix_above = suffix - s; break; }
            }
        }

        // ============ 级2 子直方图 ============
        if (need) {
            float* h2 = hist2_ + tid * HIST2_BUCKETS;
            for (int b = 0; b < HIST2_BUCKETS; ++b) h2[b] = 0.0f;
            const __m256i bev = _mm256_set1_epi32(be);
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                __m256 p = _mm256_loadu_ps(probs_ + i);
                __m256i idx = _mm256_srli_epi32(_mm256_castps_si256(p), SHIFT1);
                int m = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(idx, bev)));
                if (m) {
                    __m128 lo = _mm256_castps256_ps128(p), hi = _mm256_extractf128_ps(p, 1);
                    float f0 = _mm_cvtss_f32(lo);
                    float f1 = _mm_cvtss_f32(_mm_permute_ps(lo, 0x55));
                    float f2 = _mm_cvtss_f32(_mm_permute_ps(lo, 0xAA));
                    float f3 = _mm_cvtss_f32(_mm_permute_ps(lo, 0xFF));
                    float f4 = _mm_cvtss_f32(hi);
                    float f5 = _mm_cvtss_f32(_mm_permute_ps(hi, 0x55));
                    float f6 = _mm_cvtss_f32(_mm_permute_ps(hi, 0xAA));
                    float f7 = _mm_cvtss_f32(_mm_permute_ps(hi, 0xFF));
                    if (m & 1) h2[(f2u(f0) >> SHIFT2) & MASK2] += f0;
                    if (m & 2) h2[(f2u(f1) >> SHIFT2) & MASK2] += f1;
                    if (m & 4) h2[(f2u(f2) >> SHIFT2) & MASK2] += f2;
                    if (m & 8) h2[(f2u(f3) >> SHIFT2) & MASK2] += f3;
                    if (m & 16) h2[(f2u(f4) >> SHIFT2) & MASK2] += f4;
                    if (m & 32) h2[(f2u(f5) >> SHIFT2) & MASK2] += f5;
                    if (m & 64) h2[(f2u(f6) >> SHIFT2) & MASK2] += f6;
                    if (m & 128) h2[(f2u(f7) >> SHIFT2) & MASK2] += f7;
                }
            }
            for (; i < end; ++i) {
                uint32_t u = f2u(probs_[i]);
                if ((u >> SHIFT1) == (uint32_t)be) h2[(u >> SHIFT2) & MASK2] += probs_[i];
            }
        }
        spin_barr(3, tid, ep);

        // ============ 级2 定位 sbin + 级3/4 直方图 ============
        if (need) {
            float hs2[HIST2_BUCKETS];
            for (int b = 0; b < HIST2_BUCKETS; ++b) {
                hs2[b] = 0.0f;
                for (int t = 0; t < NT; ++t) hs2[b] += hist2_[t * HIST2_BUCKETS + b];
            }
            const float tgt1 = top_p - suffix_above;
            float suffix = 0.0f;
            for (int b = HIST2_BUCKETS - 1; b >= 0; --b) {
                suffix += hs2[b];
                if (suffix > tgt1) { sbin = b; break; }
            }
            const float s_above2 = suffix - hs2[sbin];
            const float tgt = top_p - suffix_above - s_above2;

            // 级3 + 级4 + 计数
            float* h3 = hist3_ + tid * HIST3_BUCKETS;
            float* h4 = hist4_ + tid * HIST4_BUCKETS;
            int* c4 = cnt4_ + tid * HIST4_BUCKETS;
            for (int b = 0; b < HIST3_BUCKETS; ++b) h3[b] = 0.0f;
            for (int b = 0; b < HIST4_BUCKETS; ++b) { h4[b] = 0.0f; c4[b] = 0; }
            const __m256i bev2 = _mm256_set1_epi32(be);
            const __m256i sbinv = _mm256_set1_epi32(sbin);
            const __m256i mask2v = _mm256_set1_epi32((int)MASK2);
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                __m256 p = _mm256_loadu_ps(probs_ + i);
                __m256i u = _mm256_castps_si256(p);
                __m256i hit = _mm256_and_si256(
                    _mm256_cmpeq_epi32(_mm256_srli_epi32(u, SHIFT1), bev2),
                    _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_srli_epi32(u, SHIFT2), mask2v), sbinv));
                int m = _mm256_movemask_ps(_mm256_castsi256_ps(hit));
                if (m) {
                    __m128 lo = _mm256_castps256_ps128(p), hi = _mm256_extractf128_ps(p, 1);
                    float f0 = _mm_cvtss_f32(lo);
                    float f1 = _mm_cvtss_f32(_mm_permute_ps(lo, 0x55));
                    float f2 = _mm_cvtss_f32(_mm_permute_ps(lo, 0xAA));
                    float f3 = _mm_cvtss_f32(_mm_permute_ps(lo, 0xFF));
                    float f4 = _mm_cvtss_f32(hi);
                    float f5 = _mm_cvtss_f32(_mm_permute_ps(hi, 0x55));
                    float f6 = _mm_cvtss_f32(_mm_permute_ps(hi, 0xAA));
                    float f7 = _mm_cvtss_f32(_mm_permute_ps(hi, 0xFF));
                    if (m & 1) { uint32_t uu = f2u(f0); h3[(uu >> SHIFT3) & MASK3] += f0; h4[uu & MASK4] += f0; ++c4[uu & MASK4]; }
                    if (m & 2) { uint32_t uu = f2u(f1); h3[(uu >> SHIFT3) & MASK3] += f1; h4[uu & MASK4] += f1; ++c4[uu & MASK4]; }
                    if (m & 4) { uint32_t uu = f2u(f2); h3[(uu >> SHIFT3) & MASK3] += f2; h4[uu & MASK4] += f2; ++c4[uu & MASK4]; }
                    if (m & 8) { uint32_t uu = f2u(f3); h3[(uu >> SHIFT3) & MASK3] += f3; h4[uu & MASK4] += f3; ++c4[uu & MASK4]; }
                    if (m & 16) { uint32_t uu = f2u(f4); h3[(uu >> SHIFT3) & MASK3] += f4; h4[uu & MASK4] += f4; ++c4[uu & MASK4]; }
                    if (m & 32) { uint32_t uu = f2u(f5); h3[(uu >> SHIFT3) & MASK3] += f5; h4[uu & MASK4] += f5; ++c4[uu & MASK4]; }
                    if (m & 64) { uint32_t uu = f2u(f6); h3[(uu >> SHIFT3) & MASK3] += f6; h4[uu & MASK4] += f6; ++c4[uu & MASK4]; }
                    if (m & 128) { uint32_t uu = f2u(f7); h3[(uu >> SHIFT3) & MASK3] += f7; h4[uu & MASK4] += f7; ++c4[uu & MASK4]; }
                }
            }
            for (; i < end; ++i) {
                uint32_t uu = f2u(probs_[i]);
                if ((uu >> SHIFT1) == (uint32_t)be && ((uu >> SHIFT2) & MASK2) == (uint32_t)sbin) {
                    h3[(uu >> SHIFT3) & MASK3] += probs_[i];
                    h4[uu & MASK4] += probs_[i];
                    ++c4[uu & MASK4];
                }
            }
            spin_barr(4, tid, ep);

            // 级3/4 分析（仅 tid 0）
            if (tid == 0) {
                float hs3[HIST3_BUCKETS];
                for (int b = 0; b < HIST3_BUCKETS; ++b) {
                    hs3[b] = 0.0f;
                    for (int t = 0; t < NT; ++t) hs3[b] += hist3_[t * HIST3_BUCKETS + b];
                }
                suffix = 0.0f;
                int tbin3 = 0;
                for (int b = HIST3_BUCKETS - 1; b >= 0; --b) {
                    suffix += hs3[b];
                    if (suffix >= tgt) { tbin3 = b; break; }
                }
                const float s_above3 = suffix - hs3[tbin3];
                const float tgt3 = tgt - s_above3;

                float hs4[HIST4_BUCKETS]; int hc4[HIST4_BUCKETS];
                for (int b = 0; b < HIST4_BUCKETS; ++b) {
                    hs4[b] = 0.0f; hc4[b] = 0;
                    for (int t = 0; t < NT; ++t) {
                        hs4[b] += hist4_[t * HIST4_BUCKETS + b];
                        hc4[b] += cnt4_[t * HIST4_BUCKETS + b];
                    }
                }
                suffix = 0.0f;
                int tbin4 = 0;
                for (int b = HIST4_BUCKETS - 1; b >= 0; --b) {
                    suffix += hs4[b];
                    if (suffix >= tgt3) { tbin4 = b; break; }
                }
                const float s_above4 = suffix - hs4[tbin4];

                const uint32_t th_bits = ((uint32_t)be << SHIFT1) | ((uint32_t)sbin << SHIFT2)
                                       | ((uint32_t)tbin3 << SHIFT3) | (uint32_t)tbin4;
                th_ = u2f(th_bits);
                const float gtp = s_above3 + s_above4;
                const int eqk = hc4[tbin4];
                if (th_ > 0.0f && eqk > 0) {
                    comp_ = std::min(sf((tgt - gtp) / (th_ * (float)eqk)), 1.0f);
                    comp_ = std::max(comp_, 0.0f);
                    total_ = suffix_above + s_above2 + gtp + th_ * (float)eqk * comp_;
                } else {
                    th_ = 0.0f; comp_ = 1.0f; total_ = 1.0f;
                }
            }
        } else if (tid == 0) {
            th_ = 0.0f; comp_ = 1.0f; total_ = 1.0f;
        }
        spin_barr(5, tid, ep);
        const float th = th_, comp = comp_, total = total_;

        // ============ 实例化 int 权重 + 前缀和 + 采样 ============
        const float factor = TARGET_TOTAL_F / (total > 0.0f ? total : 1.0f);
        categorical_build_weights(probs_, n, weights_, factor, th, comp, tid, NT);
        spin_barr(6, tid, ep);
        categorical_build_local_prefixes(weights_, n, prefix_sum_, block_sums_, tid, NT);
        spin_barr(7, tid, ep);
        categorical_add_local_offsets(prefix_sum_, n, block_sums_, tid, NT);
        spin_barr(8, tid, ep);
        if (tid == 0) {
            const int total_w = prefix_sum_[n - 1];
            task_.token = categorical_sample(prefix_sum_, n, total_w, next_rand(*task_.rng));
        }
    }

    void spin_barr(int stage, int tid, uint64_t ep) {
        __atomic_store_n(&arrive_[stage][tid].v[0], ep, __ATOMIC_RELEASE);
        for (int t = 0; t < NT; ++t)
            while (__atomic_load_n(&arrive_[stage][t].v[0], __ATOMIC_ACQUIRE) != ep) _mm_pause();
    }

    // ---- 全部静态缓冲 ----
    alignas(64) float probs_[MAX_N];
    alignas(64) int   weights_[MAX_N];
    alignas(64) int   prefix_sum_[MAX_N];
    alignas(64) float hist_[NT * HIST1_BUCKETS];
    alignas(64) float hist2_[NT * HIST2_BUCKETS];
    alignas(64) float hist3_[NT * HIST3_BUCKETS];
    alignas(64) float hist4_[NT * HIST4_BUCKETS];
    alignas(64) int   cnt4_[NT * HIST4_BUCKETS];
    alignas(64) float max_arr_[NT], den_arr_[NT];
    alignas(64) float th_, comp_, total_;
    alignas(64) int   block_sums_[MAX_N / 2048 + 1];
    SpinSlot arrive_[9][NT];
    uint64_t publish_ = 0;
    alignas(64) uint64_t done_[NT] = {};
    pthread_mutex_t m_;
    pthread_cond_t c_;
    pthread_t workers_[NT - 1];
    WorkerArg worker_arg_[NT - 1];
    TaskParams task_;
};

}  // namespace llm_sampling
