/*
 * rwkv_api.cpp — top-level dispatcher (fp16 + bf16 in one binary)
 * ==================================================================
 *
 * The whole engine (src/rwkv_lib.cpp + src/gemm_fp16.cpp) is compiled
 * twice in this single translation unit — once inside `namespace Fp16`
 * without RWKV_WEIGHTS_BF16 and once inside `namespace Bf16` with it.
 * The header guard macros (utils_16.h / gemv_f16.h / gemm_fp16.h) are
 * #undef'd between the two namespace blocks so the second namespace
 * re-includes them with the bf16 format selected.
 *
 * Each instance is fully format-specialized — identical codegen to the
 * former two separate builds (librwkv.so / librwkv_bf16.so).  The
 * public C API below detects the weight file magic at load time and
 * routes every call to the matching instance; the only runtime cost is
 * one perfectly-predicted branch per API call (hot loops untouched).
 */

#include "rwkv.h"

/* C++ standard headers MUST be included at global scope first:
 * including them inside a user namespace (as the engine files do when
 * embedded below) would otherwise create Fp16::std / Bf16::std and
 * break name resolution.  With the guards set here, the engine's own
 * includes are skipped inside the namespaces. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <algorithm>
#include <new>
#include <type_traits>
#include <immintrin.h>
#include <omp.h>

#define RWKV_ENGINE_EMBEDDED   /* engine files: suppress the C API */

namespace Fp16 {
#include "rwkv_lib.cpp"
#include "gemm_fp16.cpp"
#ifdef UTILS_16_H_
#undef UTILS_16_H_
#endif
#ifdef GEMV_F16_H_
#undef GEMV_F16_H_
#endif
#ifdef GEMM_FP16_H_
#undef GEMM_FP16_H_
#endif
}

namespace Bf16 {
#define RWKV_WEIGHTS_BF16
#include "rwkv_lib.cpp"
#include "gemm_fp16.cpp"
#undef RWKV_WEIGHTS_BF16
#ifdef UTILS_16_H_
#undef UTILS_16_H_
#endif
#ifdef GEMV_F16_H_
#undef GEMV_F16_H_
#endif
#ifdef GEMM_FP16_H_
#undef GEMM_FP16_H_
#endif
}

/* ── opaque handles: runtime format tag + namespaced engine instance ── */
struct RWKVWeights { int fmt; void *impl; };
struct RWKVState   { int fmt; void *impl; };

enum { FMT_FP16 = 0, FMT_BF16 = 1 };

/* gemm_fp16_set_num_threads is OpenMP-global: forward to both instances. */
void gemm_fp16_set_num_threads(int n)
{
    Fp16::gemm_fp16_set_num_threads(n);
    Bf16::gemm_fp16_set_num_threads(n);
}
int gemm_fp16_get_num_threads()
{
    return Fp16::gemm_fp16_get_num_threads();
}

extern "C" {

RWKV_API RWKVWeights *rwkv_weights_load(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return nullptr;
    char magic[7] = {0};
    size_t got = fread(magic, 1, 6, f);
    fclose(f);
    if (got < 5) return nullptr;

    RWKVWeights *w = new RWKVWeights;
    if (strncmp(magic, "RWKV7B", 6) == 0) {
        w->fmt = FMT_BF16;
        Bf16::Weights *e = new Bf16::Weights();
        if (!e->load(path)) { delete e; delete w; return nullptr; }
        w->impl = e;
    } else if (strncmp(magic, "RWKV7", 5) == 0) {
        w->fmt = FMT_FP16;
        Fp16::Weights *e = new Fp16::Weights();
        if (!e->load(path)) { delete e; delete w; return nullptr; }
        w->impl = e;
    } else {
        delete w;
        return nullptr;
    }
    return w;
}

RWKV_API void rwkv_weights_free(RWKVWeights *w)
{
    if (!w) return;
    if (w->fmt == FMT_BF16) delete (Bf16::Weights *)w->impl;
    else                    delete (Fp16::Weights *)w->impl;
    delete w;
}

/* ── Dimension accessors ─────────────────────────────────────────── */
RWKV_API int rwkv_get_dim(RWKVWeights *w)
{
    return w->fmt == FMT_BF16 ? ((Bf16::Weights *)w->impl)->dim
                              : ((Fp16::Weights *)w->impl)->dim;
}
RWKV_API int rwkv_get_n_heads(RWKVWeights *w)
{
    return w->fmt == FMT_BF16 ? ((Bf16::Weights *)w->impl)->n_heads
                              : ((Fp16::Weights *)w->impl)->n_heads;
}
RWKV_API int rwkv_get_h_size(RWKVWeights *w) { (void)w; return 64; }
RWKV_API int rwkv_get_n_layers(RWKVWeights *w)
{
    return w->fmt == FMT_BF16 ? ((Bf16::Weights *)w->impl)->n_layers
                              : ((Fp16::Weights *)w->impl)->n_layers;
}
RWKV_API int rwkv_get_vocab(RWKVWeights *w)
{
    return w->fmt == FMT_BF16 ? ((Bf16::Weights *)w->impl)->vocab
                              : ((Fp16::Weights *)w->impl)->vocab;
}
RWKV_API int rwkv_get_ffn_dim(RWKVWeights *w)
{
    return w->fmt == FMT_BF16 ? ((Bf16::Weights *)w->impl)->ffn_dim
                              : ((Fp16::Weights *)w->impl)->ffn_dim;
}

/* ── State ─────────────────────────────────────────────────────── */
RWKV_API RWKVState *rwkv_state_new(RWKVWeights *w)
{
    RWKVState *s = new RWKVState;
    s->fmt = w->fmt;
    if (w->fmt == FMT_BF16) {
        Bf16::Weights *E = (Bf16::Weights *)w->impl;
        s->impl = new Bf16::State(E->dim, E->n_heads, 64, E->n_layers);
    } else {
        Fp16::Weights *E = (Fp16::Weights *)w->impl;
        s->impl = new Fp16::State(E->dim, E->n_heads, 64, E->n_layers);
    }
    return s;
}

RWKV_API void rwkv_state_free(RWKVState *s)
{
    if (!s) return;
    if (s->fmt == FMT_BF16) delete (Bf16::State *)s->impl;
    else                    delete (Fp16::State *)s->impl;
    delete s;
}

RWKV_API void rwkv_state_reset(RWKVState *s)
{
    if (!s) return;
    if (s->fmt == FMT_BF16) ((Bf16::State *)s->impl)->reset();
    else                    ((Fp16::State *)s->impl)->reset();
}

/* ── Inference ────────────────────────────────────────────────────── */
RWKV_API void rwkv_decode(
    RWKVWeights *w, RWKVState *s,
    int token, float *logits)
{
    if (w->fmt == FMT_BF16)
        Bf16::forward_one(*(Bf16::Weights *)w->impl, *(Bf16::State *)s->impl,
                          token, logits);
    else
        Fp16::forward_one(*(Fp16::Weights *)w->impl, *(Fp16::State *)s->impl,
                          token, logits);
}

RWKV_API void rwkv_prefill(
    RWKVWeights *w, RWKVState *s,
    const int *tokens, int T,
    float *logits)
{
    if (w->fmt == FMT_BF16)
        Bf16::forward_seq(*(Bf16::Weights *)w->impl, *(Bf16::State *)s->impl,
                          tokens, T, logits);
    else
        Fp16::forward_seq(*(Fp16::Weights *)w->impl, *(Fp16::State *)s->impl,
                          tokens, T, logits);
}

/* ── Buffer sizing ─────────────────────────────────────────────────── */
/* ── Speculative decoding ─────────────────────────────────────── */
static int spec_state_size_of(RWKVWeights *w)
{
    int nl = rwkv_get_n_layers(w), dim = rwkv_get_dim(w);
    int nh = rwkv_get_n_heads(w), hs = 64;
    return (int)((size_t)nl * (2 * (size_t)dim + (size_t)nh * hs * hs));
}

RWKV_API int rwkv_state_size(RWKVWeights *w)
{
    return spec_state_size_of(w);
}

RWKV_API void rwkv_state_save(RWKVState *s, float *dst)
{
    if (!s) return;
    if (s->fmt == FMT_BF16) Bf16::state_save(*(Bf16::State *)s->impl, dst);
    else                    Fp16::state_save(*(Fp16::State *)s->impl, dst);
}

RWKV_API void rwkv_state_restore(RWKVState *s, const float *src)
{
    if (!s) return;
    if (s->fmt == FMT_BF16) Bf16::state_restore(*(Bf16::State *)s->impl, src);
    else                    Fp16::state_restore(*(Fp16::State *)s->impl, src);
}

RWKV_API void rwkv_spec_prefill(
    RWKVWeights *w, RWKVState *s,
    const int *tokens, int T,
    float *logits, float *states)
{
    if (w->fmt == FMT_BF16)
        Bf16::forward_spec(*(Bf16::Weights *)w->impl, *(Bf16::State *)s->impl,
                           tokens, T, logits, states);
    else
        Fp16::forward_spec(*(Fp16::Weights *)w->impl, *(Fp16::State *)s->impl,
                           tokens, T, logits, states);
}

/* ── Debug ─────────────────────────────────────────────────────────── */
RWKV_API void rwkv_debug_embed(RWKVWeights *w, int token, float *out)
{
    int dim = rwkv_get_dim(w);
    if (w->fmt == FMT_BF16) {
        const uint16_t *emb = ((Bf16::Weights *)w->impl)->emb;
        for (int i = 0; i < dim; i += 8)
            _mm256_storeu_ps(out + i, Bf16::load_bf16x8(emb + token * dim + i));
    } else {
        const uint16_t *emb = ((Fp16::Weights *)w->impl)->emb;
        for (int i = 0; i < dim; i += 8)
            _mm256_storeu_ps(out + i, Fp16::load_fp16x8(emb + token * dim + i));
    }
}

} // extern "C"
