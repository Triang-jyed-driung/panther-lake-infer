/*
 * fast/gemv_f16.h — Drop-in fp16 GEMV backend for RWKV-7 decode
 *
 * y[1..N] = x[1..M] · W[M][N]   where W is stored as fp16 (row-major).
 *
 * Dispatch:
 *   M*N ≤ 256×768  → single-threaded AVX2  (tiny: no OpenMP overhead)
 *   otherwise        → multi-threaded AVX2   (4 cores over output cols)
 *
 * Also provides gemv_f16_blas() for head projection (sgemm fallback).
 *
 * NOTE: no #pragma once — this header is included twice in one TU by
 * src/rwkv_api.cpp (once per format namespace, see utils_16.h), which
 * #undef-s the guard between the namespaces.
 */

#ifndef GEMV_F16_H_
#define GEMV_F16_H_
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <immintrin.h>
#include <omp.h>
#ifdef RWKV_USE_BLAS
#include <cblas.h>
#endif
#include <vector>

#include "utils_16.h"   /* to_f32 / load_16x8 — format picked at compile time */

/* ── single-threaded AVX2 GEMV ──────────────────────────────────── */
#ifdef ABLATE_GEMV
/* Scalar fallback */
static inline void gemv_f16_st(const float *x, const uint16_t *W, float *y, int M, int N) {
    memset(y, 0, N * sizeof(float));
    for (int i = 0; i < M; i++) {
        float xi = x[i];
        for (int j = 0; j < N; j++) y[j] += xi * to_f32(W[i * N + j]);
    }
}
#else
static inline void gemv_f16_st(
    const float *x, const uint16_t *W, float *y, int M, int N)
{
    memset(y, 0, N * sizeof(float));
    for (int i = 0; i < M; i++) {
        float xi = x[i];
        __m256 xv = _mm256_set1_ps(xi);
        const uint16_t *row = W + i * N;
        int j = 0;
        for (; j + 15 < N; j += 16) {
            __m256 acc0 = _mm256_loadu_ps(y + j);
            __m256 acc1 = _mm256_loadu_ps(y + j + 8);
            acc0 = _mm256_fmadd_ps(xv, load_16x8(row + j), acc0);
            acc1 = _mm256_fmadd_ps(xv, load_16x8(row + j + 8), acc1);
            _mm256_storeu_ps(y + j,     acc0);
            _mm256_storeu_ps(y + j + 8, acc1);
        }
        for (; j + 7 < N; j += 8) {
            __m256 acc = _mm256_loadu_ps(y + j);
            acc = _mm256_fmadd_ps(xv, load_16x8(row + j), acc);
            _mm256_storeu_ps(y + j, acc);
        }
        for (; j < N; j++) y[j] += xi * to_f32(row[j]);
    }
}
#endif

/* ── multi-threaded: row-partitioned (each thread owns rows, then reduce) ─ */
static inline void gemv_f16_mt_rows(
    const float *x, const uint16_t *W, float *y, int M, int N)
{
    int nth = omp_get_max_threads();
    static float *py_buf = nullptr;
    static int    py_cap = 0;
    int need = nth * N;
    if (need > py_cap) {
        free(py_buf);
        py_buf = (float *)aligned_alloc(64, need * sizeof(float));
        py_cap = need;
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int i0  = (int)((size_t)tid      * M / nth);
        int i1  = (int)((size_t)(tid + 1) * M / nth);
        float *py = py_buf + tid * N;
        for (int j = 0; j <= N - 8; j += 8) _mm256_storeu_ps(py+j,_mm256_setzero_ps());
        for (int j = (N/8)*8; j < N; j++) py[j] = 0;
        for (int i = i0; i < i1; i++) {
            __m256 xv = _mm256_set1_ps(x[i]);
            const uint16_t *row = W + i * N;
            int j = 0;
            for (; j + 15 < N; j += 16) {
                __m256 a0=_mm256_loadu_ps(py+j),a1=_mm256_loadu_ps(py+j+8);
                a0=_mm256_fmadd_ps(xv,load_16x8(row+j),a0);a1=_mm256_fmadd_ps(xv,load_16x8(row+j+8),a1);
                _mm256_storeu_ps(py+j,a0);_mm256_storeu_ps(py+j+8,a1);
            }
            for (; j + 7 < N; j += 8) {__m256 a=_mm256_loadu_ps(py+j);a=_mm256_fmadd_ps(xv,load_16x8(row+j),a);_mm256_storeu_ps(py+j,a);}
            for (; j < N; j++) py[j] += x[i] * to_f32(row[j]);
        }
    }
    #pragma omp parallel for schedule(static)
    for (int j = 0; j <= N - 8; j += 8) {
        __m256 sum = _mm256_setzero_ps();
        for (int t = 0; t < nth; t++) sum = _mm256_add_ps(sum, _mm256_loadu_ps(py_buf + t * N + j));
        _mm256_storeu_ps(y + j, sum);
    }
    for (int j = (N/8)*8; j < N; j++) {float s=0;for(int t=0;t<nth;t++)s+=py_buf[t*N+j];y[j]=s;}
}

/* ── multi-threaded: column-partitioned (each thread owns a contiguous slice of y) ─ */
static inline void gemv_f16_mt_cols(
    const float *x, const uint16_t *W, float *y, int M, int N)
{
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int j0  = (int)((size_t)tid      * N / nth);
        int j1  = (int)((size_t)(tid + 1) * N / nth);
        if (j0 >= j1)
            ; /* nothing to do for this thread */
        else {

        /* Zero our output slice (handle unaligned start + SIMD bulk + tail) */
        {
            int j = j0;
            for (; j < j1 && (j & 7); j++) y[j] = 0.0f;
            for (; j <= j1 - 8; j += 8) _mm256_storeu_ps(y + j, _mm256_setzero_ps());
            for (; j < j1; j++) y[j] = 0.0f;
        }

        /* Row-block for L1d reuse: 8 rows × (j1-j0) cols of fp16 ≤ 8×2048×2 = 32KB */
        for (int ib = 0; ib < M; ib += 8) {
            int ie = (ib + 8 < M) ? ib + 8 : M;

            int j = j0;

            /* 16-wide */
            for (; j + 15 < j1; j += 16) {
                __m256 a0 = _mm256_loadu_ps(y + j);
                __m256 a1 = _mm256_loadu_ps(y + j + 8);
                for (int i = ib; i < ie; i++) {
                    __m256 xv = _mm256_set1_ps(x[i]);
                    const uint16_t *row = W + i * N;
                    a0 = _mm256_fmadd_ps(xv, load_16x8(row + j), a0);
                    a1 = _mm256_fmadd_ps(xv, load_16x8(row + j + 8), a1);
                }
                _mm256_storeu_ps(y + j,     a0);
                _mm256_storeu_ps(y + j + 8, a1);
            }

            /* 8-wide */
            for (; j + 7 < j1; j += 8) {
                __m256 a = _mm256_loadu_ps(y + j);
                for (int i = ib; i < ie; i++)
                    a = _mm256_fmadd_ps(_mm256_set1_ps(x[i]), load_16x8(W + i * N + j), a);
                _mm256_storeu_ps(y + j, a);
            }

            /* Scalar tail */
            for (; j < j1; j++)
                for (int i = ib; i < ie; i++)
                    y[j] += x[i] * to_f32(W[i * N + j]);
        }

        } /* else j0 < j1 */
    }
}

/* ── adaptive dispatch ──────────────────────────────────────────── */
#ifdef ABLATE_GEMV
static inline void gemv_f16(const float *x, const uint16_t *W, float *y, int M, int N) {
    gemv_f16_st(x, W, y, M, N);
}
#else
static inline void gemv_f16(
    const float *x, const uint16_t *W, float *y, int M, int N)
{
    /* Small problems: single-threaded (no OpenMP overhead) */
    if ((size_t)M * N <= 256 * 768) {
        gemv_f16_st(x, W, y, M, N);
        return;
    }
    /* Column-partitioned for large N ≥ 256 (no reduction needed, direct write).
     * For very wide matrices (head, N≥8192): W should be column-major so
     * that column slices are contiguous in memory (rwkv_cpu transposes at load). */
    if (N >= 256) {
        gemv_f16_mt_cols(x, W, y, M, N);
        return;
    }
    /* Small N but large M: row-partitioned with reduction */
    gemv_f16_mt_rows(x, W, y, M, N);
}
#endif

/* ── head projection: y[V] = x[D] · W_T[V][D] ,  W_T stored [V][D] row-major ─ */
/* W_T is the transposed head weight: each ROW is a vocab entry (contiguous).
 * We parallelise over output elements — each thread computes a contiguous
 * batch of dot products, reading contiguous rows from DRAM at full bandwidth. */
static inline void gemv_f16_head(
    const float *x, const uint16_t *W_head_T, float *y,
    int D, int V)
{
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < V; j++) {
        const uint16_t *row = W_head_T + j * D;
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < D; i += 8) {
            __m256 xi = _mm256_loadu_ps(x + i);
            __m256 wi = load_16x8(row + i);
            acc = _mm256_fmadd_ps(xi, wi, acc);
        }
        /* sum256 from rwkv_cpu.cpp — declared there as inline */
        __m128 low128  = _mm256_castps256_ps128(acc);
        __m128 high128 = _mm256_extractf128_ps(acc, 1);
        __m128 sum128  = _mm_add_ps(low128, high128);
        __m128 shuf    = _mm_movehl_ps(sum128, sum128);
        sum128         = _mm_add_ps(sum128, shuf);
        shuf           = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(1, 1, 1, 1));
        sum128         = _mm_add_ps(sum128, shuf);
        y[j] = _mm_cvtss_f32(sum128);
    }
}

/* ── sparse GEMV: y[1..N] += k[1..M] · V[M][N]   (k has many zeros) ─ */
#ifdef ABLATE_SPARSE_FFN
static inline void gemv_f16_sparse(const float *k_f32, const uint16_t *V, float *y, int M, int N) {
    for (int i = 0; i < M; i++) {
        float ki = k_f32[i];
        if (ki == 0) continue;
        for (int j = 0; j < N; j++) y[j] += ki * to_f32(V[i * N + j]);
    }
}
#else
static inline void gemv_f16_sparse(
    const float *k_f32, const uint16_t *V, float *y, int M, int N)
{
    /* Phase 1: compact non-zero indices + values (single-threaded) */
    std::vector<int>   nnz_idx; nnz_idx.reserve(M);
    std::vector<float> nnz_val; nnz_val.reserve(M);

    for (int i = 0; i < M; i += 8) {
        __m256 kv  = _mm256_loadu_ps(k_f32 + i);
        __m256 cmp = _mm256_cmp_ps(kv, _mm256_setzero_ps(), _CMP_NEQ_UQ);
        uint32_t mask = _mm256_movemask_ps(cmp);
        while (mask) {
            int bit = __builtin_ctz(mask);
            nnz_idx.push_back(i + bit);
            nnz_val.push_back(k_f32[i + bit]);
            mask &= mask - 1;
        }
    }
    int nnz = (int)nnz_idx.size();

    /* Phase 2: column-parallel — each thread processes ALL nnz rows for its column slice */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int j0  = (int)((size_t)tid      * N / nth);
        int j1  = (int)((size_t)(tid + 1) * N / nth);

        for (int n = 0; n < nnz; n++) {
            int   idx = nnz_idx[n];
            float val = nnz_val[n];
            __m256 xv = _mm256_set1_ps(val);
            const uint16_t *row = V + idx * N;

            int j = j0;
            for (; j + 15 < j1; j += 16) {
                __m256 a0 = _mm256_loadu_ps(y + j);
                __m256 a1 = _mm256_loadu_ps(y + j + 8);
                a0 = _mm256_fmadd_ps(xv, load_16x8(row + j), a0);
                a1 = _mm256_fmadd_ps(xv, load_16x8(row + j + 8), a1);
                _mm256_storeu_ps(y + j,     a0);
                _mm256_storeu_ps(y + j + 8, a1);
            }
            for (; j + 7 < j1; j += 8) {
                __m256 a = _mm256_loadu_ps(y + j);
                a = _mm256_fmadd_ps(xv, load_16x8(row + j), a);
                _mm256_storeu_ps(y + j, a);
            }
            for (; j < j1; j++) y[j] += val * to_f32(row[j]);
        }
    }
}
#endif

/* ── batched GEMV for short token windows (speculative decoding) ──
 * y[t][n] = Σ_k x[t][k] · W[k][n]   for t = 0..T-1,  T ≤ 8
 *
 * W is [K][N] row-major 16-bit. Unlike the packed-GEMM path this keeps
 * all T output accumulators in YMM registers and streams W exactly
 * once — the right shape when M (=T) is tiny: no packing, no C buffer,
 * and 1× weight traffic instead of the T× of per-token GEMV calls.
 * T is a compile-time constant per instantiation so the accumulators
 * stay in registers (no spill). */
#ifdef ABLATE_GEMV
static inline void gemv_f16_batchT(
    const float *x, const uint16_t *W, float *y, int T, int K, int N)
{
    for (int t = 0; t < T; t++)
        gemv_f16_st(x + (size_t)t * K, W, y + (size_t)t * N, K, N);
}
#else
template <int TT>
static inline void gemv_batchT_cols(
    const float *x, const uint16_t *W, float *y, int K, int N)
{
    #pragma omp parallel for schedule(static)
    for (int jb = 0; jb < N; jb += 64) {
        const int j1 = (jb + 64 < N) ? jb + 64 : N;
        int j = jb;

        /* full 8-wide column blocks: TT accumulators in registers */
        for (; j + 8 <= j1; j += 8) {
            __m256 acc[TT];
            for (int t = 0; t < TT; t++) acc[t] = _mm256_setzero_ps();
            for (int i = 0; i < K; i++) {
                __m256 b = load_16x8(W + (size_t)i * N + j);
                for (int t = 0; t < TT; t++)
                    acc[t] = _mm256_fmadd_ps(
                        _mm256_set1_ps(x[(size_t)t * K + i]), b, acc[t]);
            }
            for (int t = 0; t < TT; t++)
                _mm256_storeu_ps(y + (size_t)t * N + j, acc[t]);
        }

        /* scalar tail for N % 8 */
        for (; j < j1; j++) {
            float s[TT];
            for (int t = 0; t < TT; t++) s[t] = 0.0f;
            for (int i = 0; i < K; i++) {
                float wv = to_f32(W[(size_t)i * N + j]);
                for (int t = 0; t < TT; t++) s[t] += x[(size_t)t * K + i] * wv;
            }
            for (int t = 0; t < TT; t++) y[(size_t)t * N + j] = s[t];
        }
    }
}

/* ── row-major batched GEMV for wide N (head projection) ────────────
 * gemv_batchT_cols walks W column-chunk-major: each 8-col pass strides
 * K×N bytes, so for wide matrices (N ≥ 8192, e.g. head 768×65536) the
 * 4 passes over each 64B cache line are ~K×N×2 bytes apart — every one
 * is a cold miss and the kernel runs at ~5-10 GB/s effective.
 *
 * This kernel walks W row-major instead: each thread owns a contiguous
 * column slice, accumulates into an L2-resident TT×CB y-tile, and holds
 * the accumulators in registers across KB rows. Measured (T=8):
 *   768×65536 : 3.0 ms vs 5.7-18 ms   4096×65536 : 14.6 ms vs 34 ms
 *   4096×4096 : 1.0 ms vs 4.4 ms      4096×16384 : 4.7 ms vs 5.9 ms
 * Accumulation order over k is identical → bit-equal outputs.
 * Parameters swept on 0.1B/7B head+R/K/V shapes: KB=8, CB=2048. */
template <int TT>
static inline void gemv_batchT_rows(
    const float *x, const uint16_t *W, float *y, int K, int N)
{
    const int CB = 2048;  /* y-tile columns: TT·CB·4 = 64 KB (L2) */
    const int KB = 8;     /* rows per register-accumulator block */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num(), nth = omp_get_num_threads();
        size_t c0 = (size_t)tid * N / nth, c1 = (size_t)(tid + 1) * N / nth;
        for (size_t cb = c0; cb < c1; cb += CB) {
            int C = (int)std::min<size_t>(CB, c1 - cb);
            float tile[TT][CB];
            for (int t = 0; t < TT; t++)
                for (int j = 0; j < C; j++) tile[t][j] = 0.0f;

            for (int k0 = 0; k0 < K; k0 += KB) {
                int kb = std::min(KB, K - k0);
                int j = 0;
                for (; j + 8 <= C; j += 8) {
                    __m256 acc[TT];
                    for (int t = 0; t < TT; t++)
                        acc[t] = _mm256_loadu_ps(&tile[t][j]);
                    for (int k = k0; k < k0 + kb; k++) {
                        __m256 b = load_16x8(W + (size_t)k * N + cb + j);
                        for (int t = 0; t < TT; t++)
                            acc[t] = _mm256_fmadd_ps(
                                _mm256_set1_ps(x[(size_t)t * K + k]), b, acc[t]);
                    }
                    for (int t = 0; t < TT; t++)
                        _mm256_storeu_ps(&tile[t][j], acc[t]);
                }
                for (; j < C; j++)                     /* scalar tail */
                    for (int k = k0; k < k0 + kb; k++) {
                        float wv = to_f32(W[(size_t)k * N + cb + j]);
                        for (int t = 0; t < TT; t++)
                            tile[t][j] += x[(size_t)t * K + k] * wv;
                    }
            }
            for (int t = 0; t < TT; t++)
                memcpy(y + (size_t)t * N + cb, tile[t], C * sizeof(float));
        }
    }
}

static inline void gemv_f16_batchT(
    const float *x, const uint16_t *W, float *y, int T, int K, int N)
{
    /* T == 1: batching gains nothing — use the tuned decode kernel
     * (gemv_batchT_rows<1> is FMA-latency-bound: a single accumulator
     * serializes the KB-block chain, ~3x slower than gemv_f16). */
    if (T == 1) { gemv_f16(x, W, y, K, N); return; }

    /* wide N: row-major kernel (see gemv_batchT_rows above) */
    if (N >= 4096) {
        switch (T) {
            case 1: gemv_batchT_rows<1>(x, W, y, K, N); break;
            case 2: gemv_batchT_rows<2>(x, W, y, K, N); break;
            case 3: gemv_batchT_rows<3>(x, W, y, K, N); break;
            case 4: gemv_batchT_rows<4>(x, W, y, K, N); break;
            case 5: gemv_batchT_rows<5>(x, W, y, K, N); break;
            case 6: gemv_batchT_rows<6>(x, W, y, K, N); break;
            case 7: gemv_batchT_rows<7>(x, W, y, K, N); break;
            case 8: gemv_batchT_rows<8>(x, W, y, K, N); break;
            default:
                for (int t = 0; t < T; t++)
                    gemv_f16(x + (size_t)t * K, W, y + (size_t)t * N, K, N);
        }
        return;
    }
    switch (T) {
        case 1: gemv_batchT_cols<1>(x, W, y, K, N); break;
        case 2: gemv_batchT_cols<2>(x, W, y, K, N); break;
        case 3: gemv_batchT_cols<3>(x, W, y, K, N); break;
        case 4: gemv_batchT_cols<4>(x, W, y, K, N); break;
        case 5: gemv_batchT_cols<5>(x, W, y, K, N); break;
        case 6: gemv_batchT_cols<6>(x, W, y, K, N); break;
        case 7: gemv_batchT_cols<7>(x, W, y, K, N); break;
        case 8: gemv_batchT_cols<8>(x, W, y, K, N); break;
        default:  /* T > 8: fall back to per-token GEMV */
            for (int t = 0; t < T; t++)
                gemv_f16(x + (size_t)t * K, W, y + (size_t)t * N, K, N);
    }
}
#endif /* ABLATE_GEMV */

/* ── batched sparse GEMV for the windowed FFN (ReLU² sparsity) ────
 * y[t][n] = Σ_{i: k[t][i] ≠ 0} k[t][i] · V[i][n],   T ≤ 8
 *
 * One compaction pass builds a per-row token bitmask; the column-
 * parallel pass then skips fully-zero rows entirely (≈88% of rows for
 * 0.1B) and feeds each loaded V row-block to every token that hits it,
 * keeping T output accumulators in registers. Weight traffic is the
 * UNION of the tokens' non-zero rows (served from L2/L3 across tokens)
 * instead of T full passes — the batched analogue of gemv_f16_sparse.
 * Full-store semantics (no pre-zero of y needed). */
#ifdef ABLATE_SPARSE_FFN
static inline void gemv_f16_sparse_batchT(
    const float *k, const uint16_t *V, float *y, int T, int M, int N)
{
    for (int t = 0; t < T; t++) {
        float *yt = y + (size_t)t * N;
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int i = 0; i < M; i++)
                if (k[(size_t)t * M + i] != 0.0f)
                    s += k[(size_t)t * M + i] * to_f32(V[(size_t)i * N + j]);
            yt[j] = s;
        }
    }
}
#else
template <int TT>
static inline void gemv_sparse_batchT_cols(
    const float *k, const uint16_t *V, float *y,
    int M, int N, const uint8_t *rowmask)
{
    #pragma omp parallel for schedule(static)
    for (int jb = 0; jb < N; jb += 64) {
        const int j1 = (jb + 64 < N) ? jb + 64 : N;
        int j = jb;

        for (; j + 8 <= j1; j += 8) {
            __m256 acc[TT];
            for (int t = 0; t < TT; t++) acc[t] = _mm256_setzero_ps();
            for (int i = 0; i < M; i++) {
                uint8_t m = rowmask[i];
                if (!m) continue;                     /* fully-zero row: skip */
                __m256 b = load_16x8(V + (size_t)i * N + j);
                while (m) {
                    int t = __builtin_ctz(m);
                    m &= m - 1;
                    acc[t] = _mm256_fmadd_ps(
                        _mm256_set1_ps(k[(size_t)t * M + i]), b, acc[t]);
                }
            }
            for (int t = 0; t < TT; t++)
                _mm256_storeu_ps(y + (size_t)t * N + j, acc[t]);
        }

        for (; j < j1; j++) {                         /* scalar tail */
            float s[TT];
            for (int t = 0; t < TT; t++) s[t] = 0.0f;
            for (int i = 0; i < M; i++) {
                uint8_t m = rowmask[i];
                if (!m) continue;
                float wv = to_f32(V[(size_t)i * N + j]);
                for (int t = 0; t < TT; t++)
                    if (m & (1u << t)) s[t] += k[(size_t)t * M + i] * wv;
            }
            for (int t = 0; t < TT; t++) y[(size_t)t * N + j] = s[t];
        }
    }
}

static inline void gemv_f16_sparse_batchT(
    const float *k, const uint16_t *V, float *y, int T, int M, int N)
{
    /* T == 1: use the compacting decode kernel (per-row mask scan in the
     * batched variant wastes a full M-row scalar sweep per column block). */
    if (T == 1) {
        float *yt = y;
        memset(yt, 0, N * sizeof(float));
        gemv_f16_sparse(k, V, yt, M, N);
        return;
    }

    if (T > 8) {   /* fallback: per-token sparse GEMV */
        for (int t = 0; t < T; t++) {
            float *yt = y + (size_t)t * N;
            memset(yt, 0, N * sizeof(float));
            gemv_f16_sparse(k + (size_t)t * M, V, yt, M, N);
        }
        return;
    }

    /* per-row token bitmask (single-threaded, ~M*T byte-ops — trivial) */
    static uint8_t *mask_buf = nullptr;
    static int      mask_cap = 0;
    if (M > mask_cap) {
        free(mask_buf);
        mask_buf = (uint8_t *)malloc((size_t)M);
        mask_cap = M;
    }
    memset(mask_buf, 0, (size_t)M);
    for (int t = 0; t < T; t++) {
        const float *kt = k + (size_t)t * M;
        uint8_t bit = (uint8_t)(1u << t);
        for (int i = 0; i < M; i += 8) {
            __m256 kv  = _mm256_loadu_ps(kt + i);
            __m256 cmp = _mm256_cmp_ps(kv, _mm256_setzero_ps(), _CMP_NEQ_UQ);
            uint32_t hit = _mm256_movemask_ps(cmp);
            while (hit) {
                int b = __builtin_ctz(hit);
                mask_buf[i + b] |= bit;
                hit &= hit - 1;
            }
        }
    }

    switch (T) {
        case 1: gemv_sparse_batchT_cols<1>(k, V, y, M, N, mask_buf); break;
        case 2: gemv_sparse_batchT_cols<2>(k, V, y, M, N, mask_buf); break;
        case 3: gemv_sparse_batchT_cols<3>(k, V, y, M, N, mask_buf); break;
        case 4: gemv_sparse_batchT_cols<4>(k, V, y, M, N, mask_buf); break;
        case 5: gemv_sparse_batchT_cols<5>(k, V, y, M, N, mask_buf); break;
        case 6: gemv_sparse_batchT_cols<6>(k, V, y, M, N, mask_buf); break;
        case 7: gemv_sparse_batchT_cols<7>(k, V, y, M, N, mask_buf); break;
        case 8: gemv_sparse_batchT_cols<8>(k, V, y, M, N, mask_buf); break;
    }
}
#endif /* ABLATE_SPARSE_FFN */

#ifdef RWKV_USE_BLAS
/* ── fp16→fp32 conversion + cblas_sgemm (for large N) ───────────── */
static thread_local std::vector<float> W_buf;

static void gemv_f16_blas(
    const float *x, const uint16_t *W, float *y, int M, int N)
{
    int sz = M * N;
    if ((int)W_buf.size() < sz) W_buf.resize(sz);

    /* convert fp16 → fp32 (SIMD) */
    for (int i = 0; i + 7 < sz; i += 8)
        _mm256_storeu_ps(W_buf.data() + i, load_16x8(W + i));
    for (int i = (sz / 8) * 8; i < sz; i++)
        W_buf[i] = to_f32(W[i]);

    float alpha = 1.0f, beta = 0.0f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                1, N, M, alpha, x, M, W_buf.data(), N, beta, y, N);
}
#endif

#endif /* GEMV_F16_H_ */
