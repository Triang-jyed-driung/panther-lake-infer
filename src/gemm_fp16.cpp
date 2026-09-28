// ============================================================
// FP16 GEMM — Highly Optimised AVX2+FMA+F16C Implementation
//
//   CPU target:  Core Ultra 5 325  (Arrow Lake, 4P+4E)
//   ISA:         AVX2 + FMA + F16C + AVX-VNNI  (no AVX-512)
// ============================================================

#include "gemm_fp16.h"
#ifndef RWKV_ENGINE_EMBEDDED
#include "alloc.h"
#endif
#include "utils_16.h"

#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <new>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================
// Tuning — Arrow Lake 4P+4E, L1d=40KB/core, L2=16MB(5 instances)
// ============================================================
#define MR    8
#define NR    8
#define KC  384              // fewer K-panels → less packing overhead
#define MC  128              // C-tile: 128×128×4=64KB, good for parallelism
#define NC  128              // good parallelism for N≥768

// 16-bit ↔ fp32 conversions come from utils_16.h:
//   to_f32 / load_16x8  — weight decode, fp16 or bf16 per RWKV_WEIGHTS_BF16
//   f32_to_fp16 / store_f32x8_fp16 — fp16-output legacy GEMMs

// ============================================================
// 8×8 float transpose  (AVX2)
//   r0..r7: rows → columns
//   Uses: 8 unpacks + 8 shuffles + 8 permute2f128
// ============================================================
static inline void transpose_8x8_ps(
    __m256 &r0, __m256 &r1, __m256 &r2, __m256 &r3,
    __m256 &r4, __m256 &r5, __m256 &r6, __m256 &r7)
{
    __m256 t0 = _mm256_unpacklo_ps(r0, r1);
    __m256 t1 = _mm256_unpackhi_ps(r0, r1);
    __m256 t2 = _mm256_unpacklo_ps(r2, r3);
    __m256 t3 = _mm256_unpackhi_ps(r2, r3);
    __m256 t4 = _mm256_unpacklo_ps(r4, r5);
    __m256 t5 = _mm256_unpackhi_ps(r4, r5);
    __m256 t6 = _mm256_unpacklo_ps(r6, r7);
    __m256 t7 = _mm256_unpackhi_ps(r6, r7);

    __m256 s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1,0,1,0));
    __m256 s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3,2,3,2));
    __m256 s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1,0,1,0));
    __m256 s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3,2,3,2));
    __m256 s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1,0,1,0));
    __m256 s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3,2,3,2));
    __m256 s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1,0,1,0));
    __m256 s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3,2,3,2));

    r0 = _mm256_permute2f128_ps(s0, s4, 0x20);
    r1 = _mm256_permute2f128_ps(s1, s5, 0x20);
    r2 = _mm256_permute2f128_ps(s2, s6, 0x20);
    r3 = _mm256_permute2f128_ps(s3, s7, 0x20);
    r4 = _mm256_permute2f128_ps(s0, s4, 0x31);
    r5 = _mm256_permute2f128_ps(s1, s5, 0x31);
    r6 = _mm256_permute2f128_ps(s2, s6, 0x31);
    r7 = _mm256_permute2f128_ps(s3, s7, 0x31);
}

// ============================================================
// Micro-kernel 8×8 — FP32 FMA, broadcast-based, unrolled-4
//   Interleaved accumulator ordering to hide FMA latency.
// ============================================================
static inline void ukernel_8x8(int kc,
                        const float *__restrict pa,
                        const float *__restrict pb,
                        float *__restrict c, int ldc)
{
    __m256 c0 = _mm256_loadu_ps(c + 0*ldc);
    __m256 c1 = _mm256_loadu_ps(c + 1*ldc);
    __m256 c2 = _mm256_loadu_ps(c + 2*ldc);
    __m256 c3 = _mm256_loadu_ps(c + 3*ldc);
    __m256 c4 = _mm256_loadu_ps(c + 4*ldc);
    __m256 c5 = _mm256_loadu_ps(c + 5*ldc);
    __m256 c6 = _mm256_loadu_ps(c + 6*ldc);
    __m256 c7 = _mm256_loadu_ps(c + 7*ldc);

    int k = 0;
    // 4-way unrolled main loop — interleaved for FMA latency hiding
    for (int end4 = kc - 4; k < end4; k += 4) {
        _mm_prefetch((const char *)(pa + 8*MR), _MM_HINT_T0);
        _mm_prefetch((const char *)(pb + 8*NR), _MM_HINT_T0);

        __m256 b0 = _mm256_loadu_ps(pb + 0*NR);
        __m256 b1 = _mm256_loadu_ps(pb + 1*NR);
        __m256 b2 = _mm256_loadu_ps(pb + 2*NR);
        __m256 b3 = _mm256_loadu_ps(pb + 3*NR);

        // Interleave: c0,c4,c1,c5,c2,c6,c3,c7 — each chain has 4 FMAs between uses
        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 0*MR + 0), b0, c0);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 0*MR + 4), b0, c4);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 0*MR + 1), b0, c1);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 0*MR + 5), b0, c5);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 0*MR + 2), b0, c2);
        c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 0*MR + 6), b0, c6);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 0*MR + 3), b0, c3);
        c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 0*MR + 7), b0, c7);

        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 1*MR + 0), b1, c0);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 1*MR + 4), b1, c4);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 1*MR + 1), b1, c1);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 1*MR + 5), b1, c5);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 1*MR + 2), b1, c2);
        c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 1*MR + 6), b1, c6);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 1*MR + 3), b1, c3);
        c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 1*MR + 7), b1, c7);

        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 2*MR + 0), b2, c0);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 2*MR + 4), b2, c4);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 2*MR + 1), b2, c1);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 2*MR + 5), b2, c5);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 2*MR + 2), b2, c2);
        c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 2*MR + 6), b2, c6);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 2*MR + 3), b2, c3);
        c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 2*MR + 7), b2, c7);

        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 3*MR + 0), b3, c0);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 3*MR + 4), b3, c4);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 3*MR + 1), b3, c1);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 3*MR + 5), b3, c5);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 3*MR + 2), b3, c2);
        c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 3*MR + 6), b3, c6);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 3*MR + 3), b3, c3);
        c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 3*MR + 7), b3, c7);

        pa += 4*MR;
        pb += 4*NR;
    }
    // Final tail (0-3 iterations)
    for (; k < kc; k++) {
        __m256 b0 = _mm256_loadu_ps(pb);
        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 0), b0, c0);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 1), b0, c1);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 2), b0, c2);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 3), b0, c3);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 4), b0, c4);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 5), b0, c5);
        c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 6), b0, c6);
        c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(pa + 7), b0, c7);
        pa += MR; pb += NR;
    }

    _mm256_storeu_ps(c + 0*ldc, c0);
    _mm256_storeu_ps(c + 1*ldc, c1);
    _mm256_storeu_ps(c + 2*ldc, c2);
    _mm256_storeu_ps(c + 3*ldc, c3);
    _mm256_storeu_ps(c + 4*ldc, c4);
    _mm256_storeu_ps(c + 5*ldc, c5);
    _mm256_storeu_ps(c + 6*ldc, c6);
    _mm256_storeu_ps(c + 7*ldc, c7);
}

// ============================================================
// Pack A — FP16, not transposed (A is M×K row-major)
//   8×8 transpose: read 8 f16 × 8 rows, convert, transpose, store
// ============================================================
static void packA_f16_n(int mr, int kc,
                        const unsigned short *A, int ldA,
                        int r0, int c0, float *dst)
{
    int nb = mr / MR, rem = mr % MR;
    // Full 8-row blocks
    for (int g = 0; g < nb; g++) {
        float *base = dst + g * MR * KC;
        int rs = r0 + g * MR;
        int k = 0;
        // 8-column blocks → 8×8 transpose
        for (; k + 8 <= kc; k += 8) {
            const unsigned short *col = A + rs * ldA + c0 + k;
            __m256 r0v = load_16x8(col + 0*ldA);
            __m256 r1v = load_16x8(col + 1*ldA);
            __m256 r2v = load_16x8(col + 2*ldA);
            __m256 r3v = load_16x8(col + 3*ldA);
            __m256 r4v = load_16x8(col + 4*ldA);
            __m256 r5v = load_16x8(col + 5*ldA);
            __m256 r6v = load_16x8(col + 6*ldA);
            __m256 r7v = load_16x8(col + 7*ldA);
            transpose_8x8_ps(r0v, r1v, r2v, r3v, r4v, r5v, r6v, r7v);
            _mm256_storeu_ps(base + (k+0)*MR, r0v);
            _mm256_storeu_ps(base + (k+1)*MR, r1v);
            _mm256_storeu_ps(base + (k+2)*MR, r2v);
            _mm256_storeu_ps(base + (k+3)*MR, r3v);
            _mm256_storeu_ps(base + (k+4)*MR, r4v);
            _mm256_storeu_ps(base + (k+5)*MR, r5v);
            _mm256_storeu_ps(base + (k+6)*MR, r6v);
            _mm256_storeu_ps(base + (k+7)*MR, r7v);
        }
        // Remainder columns (scalar)
        for (; k < kc; k++)
            for (int r = 0; r < MR; r++)
                base[k * MR + r] = to_f32(A[(rs + r) * ldA + c0 + k]);
    }
    // Remainder rows (scalar)
    if (rem) {
        float *base = dst + nb * MR * KC;
        int rs = r0 + nb * MR;
        for (int r = 0; r < rem; r++) {
            const unsigned short *row = A + (rs + r) * ldA + c0;
            for (int k = 0; k < kc; k++)
                base[k * MR + r] = to_f32(row[k]);
        }
    }
}

// Pack A — FP16, transposed (A is K×M row-major, rows = K, cols = M)
//   Each "row" of A is contiguous in the M dimension → direct load
static void packA_f16_t(int mr, int kc,
                        const unsigned short *A, int ldA,
                        int r0, int c0, float *dst)
{
    int nb = mr / MR, rem = mr % MR;
    for (int g = 0; g < nb; g++) {
        float *base = dst + g * MR * KC;
        int rs = r0 + g * MR;
        for (int k = 0; k < kc; k++)
            _mm256_storeu_ps(base + k * MR,
                load_16x8(A + (c0 + k) * ldA + rs));
    }
    if (rem) {
        float *base = dst + nb * MR * KC;
        int rs = r0 + nb * MR;
        for (int k = 0; k < kc; k++)
            for (int r = 0; r < rem; r++)
                base[k * MR + r] = to_f32(A[(c0 + k) * ldA + rs + r]);
    }
}

// ============================================================
// Pack A — FP32, not transposed
//   8×8 transpose: read 8 f32 × 8 rows, transpose, store
// ============================================================
static void packA_f32_n(int mr, int kc,
                        const float *A, int ldA,
                        int r0, int c0, float *dst)
{
    int nb = mr / MR, rem = mr % MR;
    for (int g = 0; g < nb; g++) {
        float *base = dst + g * MR * KC;
        int rs = r0 + g * MR;
        int k = 0;
        for (; k + 8 <= kc; k += 8) {
            const float *col = A + rs * ldA + c0 + k;
            __m256 r0v = _mm256_loadu_ps(col + 0*ldA);
            __m256 r1v = _mm256_loadu_ps(col + 1*ldA);
            __m256 r2v = _mm256_loadu_ps(col + 2*ldA);
            __m256 r3v = _mm256_loadu_ps(col + 3*ldA);
            __m256 r4v = _mm256_loadu_ps(col + 4*ldA);
            __m256 r5v = _mm256_loadu_ps(col + 5*ldA);
            __m256 r6v = _mm256_loadu_ps(col + 6*ldA);
            __m256 r7v = _mm256_loadu_ps(col + 7*ldA);
            transpose_8x8_ps(r0v, r1v, r2v, r3v, r4v, r5v, r6v, r7v);
            _mm256_storeu_ps(base + (k+0)*MR, r0v);
            _mm256_storeu_ps(base + (k+1)*MR, r1v);
            _mm256_storeu_ps(base + (k+2)*MR, r2v);
            _mm256_storeu_ps(base + (k+3)*MR, r3v);
            _mm256_storeu_ps(base + (k+4)*MR, r4v);
            _mm256_storeu_ps(base + (k+5)*MR, r5v);
            _mm256_storeu_ps(base + (k+6)*MR, r6v);
            _mm256_storeu_ps(base + (k+7)*MR, r7v);
        }
        for (; k < kc; k++)
            for (int r = 0; r < MR; r++)
                base[k * MR + r] = A[(rs + r) * ldA + c0 + k];
    }
    if (rem) {
        float *base = dst + nb * MR * KC;
        int rs = r0 + nb * MR;
        for (int r = 0; r < rem; r++) {
            const float *row = A + (rs + r) * ldA + c0;
            for (int k = 0; k < kc; k++)
                base[k * MR + r] = row[k];
        }
    }
}

// Pack A — FP32, transposed (A is K×M)
static void packA_f32_t(int mr, int kc,
                        const float *A, int ldA,
                        int r0, int c0, float *dst)
{
    int nb = mr / MR, rem = mr % MR;
    for (int g = 0; g < nb; g++) {
        float *base = dst + g * MR * KC;
        int rs = r0 + g * MR;
        for (int k = 0; k < kc; k++)
            _mm256_storeu_ps(base + k * MR,
                _mm256_loadu_ps(A + (c0 + k) * ldA + rs));
    }
    if (rem) {
        float *base = dst + nb * MR * KC;
        int rs = r0 + nb * MR;
        for (int k = 0; k < kc; k++)
            for (int r = 0; r < rem; r++)
                base[k * MR + r] = A[(c0 + k) * ldA + rs + r];
    }
}

// ============================================================
// Pack B — FP16, not transposed (B is K×N row-major)
//   Each row of B is contiguous in N → direct vector load
// ============================================================
static void packB_f16_n(int kc, int nr,
                        const unsigned short *B, int ldB,
                        int r0, int c0, float *dst)
{
    int nb = nr / NR, rem = nr % NR;
    for (int g = 0; g < nb; g++) {
        float *base = dst + g * KC * NR;
        int cs = c0 + g * NR;
        for (int k = 0; k < kc; k++)
            _mm256_storeu_ps(base + k * NR,
                load_16x8(B + (r0 + k) * ldB + cs));
    }
    if (rem) {
        float *base = dst + nb * KC * NR;
        int cs = c0 + nb * NR;
        for (int k = 0; k < kc; k++)
            for (int c = 0; c < rem; c++)
                base[k * NR + c] = to_f32(B[(r0 + k) * ldB + cs + c]);
    }
}

// Pack B — FP16, transposed (B is N×K row-major = original K×N, transposed)
//   8×8 transpose: read 8 f16 × 8 "rows" (columns of original B), convert, transpose
static void packB_f16_t(int kc, int nr,
                        const unsigned short *B, int ldB,
                        int r0, int c0, float *dst)
{
    int nb = nr / NR, rem = nr % NR;
    for (int g = 0; g < nb; g++) {
        float *base = dst + g * KC * NR;
        int cs = c0 + g * NR;
        int k = 0;
        for (; k + 8 <= kc; k += 8) {
            // Load 8 f16 from each of 8 "rows" of transposed B
            // Row j (j = cs+0..cs+7) at k-offset k..k+7
            __m256 r0v = load_16x8(B + (cs+0)*ldB + r0 + k);
            __m256 r1v = load_16x8(B + (cs+1)*ldB + r0 + k);
            __m256 r2v = load_16x8(B + (cs+2)*ldB + r0 + k);
            __m256 r3v = load_16x8(B + (cs+3)*ldB + r0 + k);
            __m256 r4v = load_16x8(B + (cs+4)*ldB + r0 + k);
            __m256 r5v = load_16x8(B + (cs+5)*ldB + r0 + k);
            __m256 r6v = load_16x8(B + (cs+6)*ldB + r0 + k);
            __m256 r7v = load_16x8(B + (cs+7)*ldB + r0 + k);
            transpose_8x8_ps(r0v, r1v, r2v, r3v, r4v, r5v, r6v, r7v);
            _mm256_storeu_ps(base + (k+0)*NR, r0v);
            _mm256_storeu_ps(base + (k+1)*NR, r1v);
            _mm256_storeu_ps(base + (k+2)*NR, r2v);
            _mm256_storeu_ps(base + (k+3)*NR, r3v);
            _mm256_storeu_ps(base + (k+4)*NR, r4v);
            _mm256_storeu_ps(base + (k+5)*NR, r5v);
            _mm256_storeu_ps(base + (k+6)*NR, r6v);
            _mm256_storeu_ps(base + (k+7)*NR, r7v);
        }
        for (; k < kc; k++)
            for (int c = 0; c < NR; c++)
                base[k * NR + c] = to_f32(B[(cs + c) * ldB + r0 + k]);
    }
    if (rem) {
        float *base = dst + nb * KC * NR;
        int cs = c0 + nb * NR;
        for (int k = 0; k < kc; k++)
            for (int c = 0; c < rem; c++)
                base[k * NR + c] = to_f32(B[(cs + c) * ldB + r0 + k]);
    }
}

// ============================================================
// Pack B — FP32, not transposed
// ============================================================
static void packB_f32_n(int kc, int nr,
                        const float *B, int ldB,
                        int r0, int c0, float *dst)
{
    int nb = nr / NR, rem = nr % NR;
    for (int g = 0; g < nb; g++) {
        float *base = dst + g * KC * NR;
        int cs = c0 + g * NR;
        for (int k = 0; k < kc; k++)
            _mm256_storeu_ps(base + k * NR,
                _mm256_loadu_ps(B + (r0 + k) * ldB + cs));
    }
    if (rem) {
        float *base = dst + nb * KC * NR;
        int cs = c0 + nb * NR;
        for (int k = 0; k < kc; k++)
            for (int c = 0; c < rem; c++)
                base[k * NR + c] = B[(r0 + k) * ldB + cs + c];
    }
}

// Pack B — FP32, transposed (B is N×K row-major)
//   8×8 transpose
static void packB_f32_t(int kc, int nr,
                        const float *B, int ldB,
                        int r0, int c0, float *dst)
{
    int nb = nr / NR, rem = nr % NR;
    for (int g = 0; g < nb; g++) {
        float *base = dst + g * KC * NR;
        int cs = c0 + g * NR;
        int k = 0;
        for (; k + 8 <= kc; k += 8) {
            __m256 r0v = _mm256_loadu_ps(B + (cs+0)*ldB + r0 + k);
            __m256 r1v = _mm256_loadu_ps(B + (cs+1)*ldB + r0 + k);
            __m256 r2v = _mm256_loadu_ps(B + (cs+2)*ldB + r0 + k);
            __m256 r3v = _mm256_loadu_ps(B + (cs+3)*ldB + r0 + k);
            __m256 r4v = _mm256_loadu_ps(B + (cs+4)*ldB + r0 + k);
            __m256 r5v = _mm256_loadu_ps(B + (cs+5)*ldB + r0 + k);
            __m256 r6v = _mm256_loadu_ps(B + (cs+6)*ldB + r0 + k);
            __m256 r7v = _mm256_loadu_ps(B + (cs+7)*ldB + r0 + k);
            transpose_8x8_ps(r0v, r1v, r2v, r3v, r4v, r5v, r6v, r7v);
            _mm256_storeu_ps(base + (k+0)*NR, r0v);
            _mm256_storeu_ps(base + (k+1)*NR, r1v);
            _mm256_storeu_ps(base + (k+2)*NR, r2v);
            _mm256_storeu_ps(base + (k+3)*NR, r3v);
            _mm256_storeu_ps(base + (k+4)*NR, r4v);
            _mm256_storeu_ps(base + (k+5)*NR, r5v);
            _mm256_storeu_ps(base + (k+6)*NR, r6v);
            _mm256_storeu_ps(base + (k+7)*NR, r7v);
        }
        for (; k < kc; k++)
            for (int c = 0; c < NR; c++)
                base[k * NR + c] = B[(cs + c) * ldB + r0 + k];
    }
    if (rem) {
        float *base = dst + nb * KC * NR;
        int cs = c0 + nb * NR;
        for (int k = 0; k < kc; k++)
            for (int c = 0; c < rem; c++)
                base[k * NR + c] = B[(cs + c) * ldB + r0 + k];
    }
}

// ============================================================
// Convert FP32 → FP16  (vectorised output)
// ============================================================
static void cvt_f32_to_f16(int rows, int cols,
                           const float *src, int lds,
                           unsigned short *dst, int ldd)
{
    for (int r = 0; r < rows; r++) {
        const float *sr = src + r * lds;
        unsigned short *dr = dst + r * ldd;
        int c = 0;
        for (; c + 32 <= cols; c += 32) {
            __m128i h0 = _mm256_cvtps_ph(_mm256_loadu_ps(sr + c +  0), _MM_FROUND_TO_NEAREST_INT);
            __m128i h1 = _mm256_cvtps_ph(_mm256_loadu_ps(sr + c +  8), _MM_FROUND_TO_NEAREST_INT);
            __m128i h2 = _mm256_cvtps_ph(_mm256_loadu_ps(sr + c + 16), _MM_FROUND_TO_NEAREST_INT);
            __m128i h3 = _mm256_cvtps_ph(_mm256_loadu_ps(sr + c + 24), _MM_FROUND_TO_NEAREST_INT);
            _mm256_storeu_si256((__m256i *)(dr + c),
                _mm256_insertf128_si256(_mm256_castsi128_si256(h0), h1, 1));
            _mm256_storeu_si256((__m256i *)(dr + c + 16),
                _mm256_insertf128_si256(_mm256_castsi128_si256(h2), h3, 1));
        }
        for (; c + 8 <= cols; c += 8)
            _mm_storeu_si128((__m128i *)(dr + c),
                _mm256_cvtps_ph(_mm256_loadu_ps(sr + c), _MM_FROUND_TO_NEAREST_INT));
        for (; c < cols; c++)
            dr[c] = f32_to_fp16(sr[c]);
    }
}

// ============================================================
// Per-element accessor helpers for scalar remainder
//   Type-dispatch via template, so no runtime branching
// ============================================================
template <typename T>
static inline float elem(const T *ptr, int i) { return (float)ptr[i]; }
template <>
inline float elem<unsigned short>(const unsigned short *ptr, int i) { return to_f32(ptr[i]); }

template <typename TA, typename TB>
static inline float getA(const TA *A, int ldA, bool A_t, int row, int k) {
    return A_t ? elem<TA>(A, k * ldA + row) : elem<TA>(A, row * ldA + k);
}
template <typename TA, typename TB>
static inline float getB(const TB *B, int ldB, bool B_t, int k, int col) {
    return B_t ? elem<TB>(B, col * ldB + k) : elem<TB>(B, k * ldB + col);
}

// ============================================================
// Core GEMM loop
//
// Template parameters encode the input types for the remainder path.
// packA / packB are function pointers already matching the types.
// ============================================================
typedef void (*packA_fn)(int mr, int kc, const void *A, int ldA, int r0, int c0, float *dst);
typedef void (*packB_fn)(int kc, int nr, const void *B, int ldB, int r0, int c0, float *dst);

template <typename TA, typename TB, bool isFp16Out>
static void gemm_core(int M, int N, int K,
                      const TA *A, int ldA, bool A_trans, packA_fn packA,
                      const TB *B, int ldB, bool B_trans, packB_fn packB,
                      void *C_out, int ldC)
{
    int MxN = M * N;
    AlignedBuffer C_buf(MxN);
    float *C_f32 = C_buf.data();
    if (!C_f32) return;

#pragma omp parallel
    {
        THREAD_LOCAL_BUFFER(bufA, MC * KC);
        THREAD_LOCAL_BUFFER(bufB, KC * NC);
        // Per-tile local accumulation buffer to eliminate false sharing on C
        THREAD_LOCAL_BUFFER(tileC, MC * NC);
        if (!bufA || !bufB || !tileC) { 
            // 分配失败，跳过
        } else {
#pragma omp for schedule(dynamic, 1) collapse(2) nowait
            for (int ic = 0; ic < M; ic += MC) {
                for (int jc = 0; jc < N; jc += NC) {
                    int mc = std::min(MC, M - ic);
                    int nc = std::min(NC, N - jc);
                    int mc_full = (mc / MR) * MR;
                    int nc_full = (nc / NR) * NR;

                    // Zero local tile buffer
                    memset(tileC, 0, mc * nc * sizeof(float));

                    for (int pc = 0; pc < K; pc += KC) {
                        int kc = std::min(KC, K - pc);

                        // Pack B for all NR blocks
                        if (nc_full > 0) packB(kc, nc_full, B, ldB, pc, jc, bufB);

                        // Full 8×8 micro-kernel — writes to tileC (no false sharing)
                        if (mc_full > 0 && nc_full > 0) {
                            packA(mc_full, kc, A, ldA, ic, pc, bufA);
                            int n_nr = nc_full / NR;
                            int n_mr = mc_full / MR;
                            for (int jr = 0; jr < n_nr; jr++) {
                                const float *pb = bufB + jr * KC * NR;
                                for (int ir = 0; ir < n_mr; ir++) {
                                    const float *pa = bufA + ir * MR * KC;
                                    float *csub = tileC + (ir * MR) * nc + (jr * NR);
                                    ukernel_8x8(kc, pa, pb, csub, nc);
                                }
                            }
                        }

                        // Scalar remainder: only the leftover edges
                        if (mc > mc_full || nc > nc_full) {
                            for (int i = ic; i < ic + mc_full; i++) {
                                for (int j = jc + nc_full; j < jc + nc; j++) {
                                    float acc = 0;
                                    for (int k = 0; k < kc; k++)
                                        acc += getA<TA,TB>(A, ldA, A_trans, i, pc+k)
                                             * getB<TA,TB>(B, ldB, B_trans, pc+k, j);
                                    tileC[(i-ic) * nc + (j-jc)] += acc;
                                }
                            }
                            for (int i = ic + mc_full; i < ic + mc; i++) {
                                for (int j = jc; j < jc + nc; j++) {
                                    float acc = 0;
                                    for (int k = 0; k < kc; k++)
                                        acc += getA<TA,TB>(A, ldA, A_trans, i, pc+k)
                                             * getB<TA,TB>(B, ldB, B_trans, pc+k, j);
                                    tileC[(i-ic) * nc + (j-jc)] += acc;
                                }
                            }
                        }
                    } // pc

                    // Write tile back to global C (once, no contention)
                    for (int i = 0; i < mc; i++)
                        memcpy(C_f32 + (ic + i) * N + jc, tileC + i * nc, nc * sizeof(float));
                } // jc
            } // ic
            // C_buf 自动释放
        }
    }

    // Output
    if (isFp16Out)
        cvt_f32_to_f16(M, N, C_f32, N, (unsigned short *)C_out, ldC);
    else {
        float *out = (float *)C_out;
        for (int i = 0; i < M; i++)
            memcpy(out + i * ldC, C_f32 + i * N, N * sizeof(float));
    }
}

// ============================================================
// Dispatch wrappers  (all 8 type combinations)
// ============================================================
// 1. f16,f16→f16
static void fp16_gemm_impl(size_t M, size_t N, size_t K,
               const unsigned short *A, const unsigned short *B,
               unsigned short *C, bool A_t, bool B_t)
{
    if (!M||!N||!K) return;
    gemm_core<unsigned short, unsigned short, true>(
        (int)M,(int)N,(int)K,
        A, A_t?(int)M:(int)K, A_t,
        A_t?(packA_fn)packA_f16_t:(packA_fn)packA_f16_n,
        B, B_t?(int)K:(int)N, B_t,
        B_t?(packB_fn)packB_f16_t:(packB_fn)packB_f16_n,
        C,(int)N);
}
// 2. f32,f16→f16
static void fp32_fp16_gemm_impl(size_t M, size_t N, size_t K,
                    const float *A, const unsigned short *B,
                    unsigned short *C, bool A_t, bool B_t)
{
    if (!M||!N||!K) return;
    gemm_core<float, unsigned short, true>(
        (int)M,(int)N,(int)K,
        A, A_t?(int)M:(int)K, A_t,
        A_t?(packA_fn)packA_f32_t:(packA_fn)packA_f32_n,
        B, B_t?(int)K:(int)N, B_t,
        B_t?(packB_fn)packB_f16_t:(packB_fn)packB_f16_n,
        C,(int)N);
}
// 3. f16,f32→f16
static void fp16_fp32_gemm_impl(size_t M, size_t N, size_t K,
                    const unsigned short *A, const float *B,
                    unsigned short *C, bool A_t, bool B_t)
{
    if (!M||!N||!K) return;
    gemm_core<unsigned short, float, true>(
        (int)M,(int)N,(int)K,
        A, A_t?(int)M:(int)K, A_t,
        A_t?(packA_fn)packA_f16_t:(packA_fn)packA_f16_n,
        B, B_t?(int)K:(int)N, B_t,
        B_t?(packB_fn)packB_f32_t:(packB_fn)packB_f32_n,
        C,(int)N);
}
// 4. f16,f16→f32
static void fp16_fp16_fp32_gemm_impl(size_t M, size_t N, size_t K,
                    const unsigned short *A, const unsigned short *B,
                    float *C, bool A_t, bool B_t)
{
    if (!M||!N||!K) return;
    gemm_core<unsigned short, unsigned short, false>(
        (int)M,(int)N,(int)K,
        A, A_t?(int)M:(int)K, A_t,
        A_t?(packA_fn)packA_f16_t:(packA_fn)packA_f16_n,
        B, B_t?(int)K:(int)N, B_t,
        B_t?(packB_fn)packB_f16_t:(packB_fn)packB_f16_n,
        C,(int)N);
}
// 6x16 AVX2 kernel: twelve independent accumulators amortize broadcasts
// across two vectors. B is converted once per K/N panel and reused by all
// token rows. A stays in its original FP32 layout (no repeated A packing).
static inline void kernel_6x16(int kc, const float *a, int lda,
                              const float *b, float *c, int ldc) {
    __m256 c00 = _mm256_loadu_ps(c + 0 * ldc + 0);
    __m256 c01 = _mm256_loadu_ps(c + 0 * ldc + 8);
    __m256 c10 = _mm256_loadu_ps(c + 1 * ldc + 0);
    __m256 c11 = _mm256_loadu_ps(c + 1 * ldc + 8);
    __m256 c20 = _mm256_loadu_ps(c + 2 * ldc + 0);
    __m256 c21 = _mm256_loadu_ps(c + 2 * ldc + 8);
    __m256 c30 = _mm256_loadu_ps(c + 3 * ldc + 0);
    __m256 c31 = _mm256_loadu_ps(c + 3 * ldc + 8);
    __m256 c40 = _mm256_loadu_ps(c + 4 * ldc + 0);
    __m256 c41 = _mm256_loadu_ps(c + 4 * ldc + 8);
    __m256 c50 = _mm256_loadu_ps(c + 5 * ldc + 0);
    __m256 c51 = _mm256_loadu_ps(c + 5 * ldc + 8);
    for (int k = 0; k < kc; ++k) {
        __m256 b0 = _mm256_loadu_ps(b + k * 16);
        __m256 b1 = _mm256_loadu_ps(b + k * 16 + 8);
        __m256 a0 = _mm256_broadcast_ss(a + 0 * lda + k);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        __m256 a1 = _mm256_broadcast_ss(a + 1 * lda + k);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        __m256 a2 = _mm256_broadcast_ss(a + 2 * lda + k);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        __m256 a3 = _mm256_broadcast_ss(a + 3 * lda + k);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
        __m256 a4 = _mm256_broadcast_ss(a + 4 * lda + k);
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);
        __m256 a5 = _mm256_broadcast_ss(a + 5 * lda + k);
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);
    }
    _mm256_storeu_ps(c + 0 * ldc + 0, c00);
    _mm256_storeu_ps(c + 0 * ldc + 8, c01);
    _mm256_storeu_ps(c + 1 * ldc + 0, c10);
    _mm256_storeu_ps(c + 1 * ldc + 8, c11);
    _mm256_storeu_ps(c + 2 * ldc + 0, c20);
    _mm256_storeu_ps(c + 2 * ldc + 8, c21);
    _mm256_storeu_ps(c + 3 * ldc + 0, c30);
    _mm256_storeu_ps(c + 3 * ldc + 8, c31);
    _mm256_storeu_ps(c + 4 * ldc + 0, c40);
    _mm256_storeu_ps(c + 4 * ldc + 8, c41);
    _mm256_storeu_ps(c + 5 * ldc + 0, c50);
    _mm256_storeu_ps(c + 5 * ldc + 8, c51);
}

static void gemm_prefill(int M, int N, int K, const float *A,
                         const unsigned short *B, float *C) {
    constexpr int BK = 256;
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    // Balance narrow gate projections and 768-wide layers across the team.
    // Fixed 128-column tiles leave three cores idle for N=128 and distribute
    // six tiles unevenly for N=768 on four cores.
    int panels = (N + 127) / 128;
    panels = ((panels + threads - 1) / threads) * threads;
    const int BN = ((N + panels * 16 - 1) / (panels * 16)) * 16;
    // Preserve the API's in-place activation projection support (A == C).
    AlignedBuffer output((size_t)M * N);
    float *out = output.data();
    #pragma omp parallel
    {
        THREAD_LOCAL_BUFFER(packed, BK * 128);
        #pragma omp for schedule(static)
        for (int jc = 0; jc < N; jc += BN) {
            int nc = std::min(BN, N - jc);
            for (int i = 0; i < M; ++i)
                memset(out + (size_t)i * N + jc, 0, nc * sizeof(float));
            for (int pc = 0; pc < K; pc += BK) {
                int kc = std::min(BK, K - pc);
                // Read complete source cache lines together, avoiding the
                // repeated strided sweeps of an eight-column packer.
                for (int k = 0; k < kc; ++k)
                    for (int j = 0; j < nc; j += 16) {
                        float *dst = packed + (j / 16) * BK * 16 + k * 16;
                        const unsigned short *src = B + (size_t)(pc + k) * N + jc + j;
                        _mm256_storeu_ps(dst, load_16x8(src));
                        _mm256_storeu_ps(dst + 8, load_16x8(src + 8));
                    }
                for (int j = 0; j < nc; j += 16) {
                    const float *pb = packed + (j / 16) * BK * 16;
                    int i = 0;
                    for (; i + 6 <= M; i += 6)
                        kernel_6x16(kc, A + (size_t)i * K + pc, K,
                                    pb, out + (size_t)i * N + jc + j, N);
                    for (; i < M; ++i) {
                        float *dst = out + (size_t)i * N + jc + j;
                        __m256 c0 = _mm256_loadu_ps(dst);
                        __m256 c1 = _mm256_loadu_ps(dst + 8);
                        for (int k = 0; k < kc; ++k) {
                            __m256 a = _mm256_set1_ps(A[(size_t)i * K + pc + k]);
                            c0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(pb + k * 16), c0);
                            c1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(pb + k * 16 + 8), c1);
                        }
                        _mm256_storeu_ps(dst, c0);
                        _mm256_storeu_ps(dst + 8, c1);
                    }
                }
            }
        }
    }
    memcpy(C, out, (size_t)M * N * sizeof(float));
}

// 5. f32,f16→f32
static void fp32_fp16_fp32_gemm_impl(size_t M, size_t N, size_t K,
                    const float *A, const unsigned short *B,
                    float *C, bool A_t, bool B_t)
{
    if (!M||!N||!K) return;
    if (!A_t && !B_t && N % 16 == 0 && M >= 6) {
        gemm_prefill((int)M, (int)N, (int)K, A, B, C);
        return;
    }

    gemm_core<float, unsigned short, false>(
        (int)M,(int)N,(int)K,
        A, A_t?(int)M:(int)K, A_t,
        A_t?(packA_fn)packA_f32_t:(packA_fn)packA_f32_n,
        B, B_t?(int)K:(int)N, B_t,
        B_t?(packB_fn)packB_f16_t:(packB_fn)packB_f16_n,
        C,(int)N);
}
// 6. f16,f32→f32
static void fp16_fp32_fp32_gemm_impl(size_t M, size_t N, size_t K,
                    const unsigned short *A, const float *B,
                    float *C, bool A_t, bool B_t)
{
    if (!M||!N||!K) return;
    gemm_core<unsigned short, float, false>(
        (int)M,(int)N,(int)K,
        A, A_t?(int)M:(int)K, A_t,
        A_t?(packA_fn)packA_f16_t:(packA_fn)packA_f16_n,
        B, B_t?(int)K:(int)N, B_t,
        B_t?(packB_fn)packB_f32_t:(packB_fn)packB_f32_n,
        C,(int)N);
}
// 7. f32,f32→f16
static void fp32_fp32_fp16_gemm_impl(size_t M, size_t N, size_t K,
                    const float *A, const float *B,
                    unsigned short *C, bool A_t, bool B_t)
{
    if (!M||!N||!K) return;
    gemm_core<float, float, true>(
        (int)M,(int)N,(int)K,
        A, A_t?(int)M:(int)K, A_t,
        A_t?(packA_fn)packA_f32_t:(packA_fn)packA_f32_n,
        B, B_t?(int)K:(int)N, B_t,
        B_t?(packB_fn)packB_f32_t:(packB_fn)packB_f32_n,
        C,(int)N);
}
// 8. f32,f32→f32
static void fp32_gemm_impl(size_t M, size_t N, size_t K,
                      const float *A, const float *B, float *C, bool A_t, bool B_t)
{
    if (!M||!N||!K) return;
    gemm_core<float, float, false>(
        (int)M,(int)N,(int)K,
        A, A_t?(int)M:(int)K, A_t,
        A_t?(packA_fn)packA_f32_t:(packA_fn)packA_f32_n,
        B, B_t?(int)K:(int)N, B_t,
        B_t?(packB_fn)packB_f32_t:(packB_fn)packB_f32_n,
        C,(int)N);
}

// ============================================================
// Public API
// ============================================================
void fp16_gemm(size_t M, size_t N, size_t K,
               const unsigned short *A, const unsigned short *B,
               unsigned short *C, bool A_t, bool B_t)
{ fp16_gemm_impl(M,N,K,A,B,C,A_t,B_t); }

void fp32_fp16_gemm(size_t M, size_t N, size_t K,
                    const float *A, const unsigned short *B,
                    unsigned short *C, bool A_t, bool B_t)
{ fp32_fp16_gemm_impl(M,N,K,A,B,C,A_t,B_t); }

void fp16_fp32_gemm(size_t M, size_t N, size_t K,
                    const unsigned short *A, const float *B,
                    unsigned short *C, bool A_t, bool B_t)
{ fp16_fp32_gemm_impl(M,N,K,A,B,C,A_t,B_t); }

// ============================================================
// Template interface
// ============================================================
template <typename T>
static constexpr bool is_f16 = std::is_same<T,unsigned short>::value || std::is_same<T,uint16_t>::value;

template <typename TA, typename TB, typename TC>
void gemm(size_t M, size_t N, size_t K, const TA *A, const TB *B, TC *C, bool A_t, bool B_t)
{
    if constexpr (is_f16<TA> && is_f16<TB> && is_f16<TC>)
        fp16_gemm_impl(M,N,K,(const unsigned short*)A,(const unsigned short*)B,(unsigned short*)C,A_t,B_t);
    else if constexpr (!is_f16<TA> && is_f16<TB> && is_f16<TC>)
        fp32_fp16_gemm_impl(M,N,K,(const float*)A,(const unsigned short*)B,(unsigned short*)C,A_t,B_t);
    else if constexpr (is_f16<TA> && !is_f16<TB> && is_f16<TC>)
        fp16_fp32_gemm_impl(M,N,K,(const unsigned short*)A,(const float*)B,(unsigned short*)C,A_t,B_t);
    else if constexpr (is_f16<TA> && is_f16<TB> && !is_f16<TC>)
        fp16_fp16_fp32_gemm_impl(M,N,K,(const unsigned short*)A,(const unsigned short*)B,(float*)C,A_t,B_t);
    else if constexpr (!is_f16<TA> && is_f16<TB> && !is_f16<TC>)
        fp32_fp16_fp32_gemm_impl(M,N,K,(const float*)A,(const unsigned short*)B,(float*)C,A_t,B_t);
    else if constexpr (is_f16<TA> && !is_f16<TB> && !is_f16<TC>)
        fp16_fp32_fp32_gemm_impl(M,N,K,(const unsigned short*)A,(const float*)B,(float*)C,A_t,B_t);
    else if constexpr (!is_f16<TA> && !is_f16<TB> && is_f16<TC>)
        fp32_fp32_fp16_gemm_impl(M,N,K,(const float*)A,(const float*)B,(unsigned short*)C,A_t,B_t);
    else if constexpr (!is_f16<TA> && !is_f16<TB> && !is_f16<TC>)
        fp32_gemm_impl(M,N,K,(const float*)A,(const float*)B,(float*)C,A_t,B_t);
}

// Explicit instantiations: all 8 type combinations
template void gemm<unsigned short,unsigned short,unsigned short>(size_t,size_t,size_t,const unsigned short*,const unsigned short*,unsigned short*,bool,bool);
template void gemm<float,unsigned short,unsigned short>(size_t,size_t,size_t,const float*,const unsigned short*,unsigned short*,bool,bool);
template void gemm<unsigned short,float,unsigned short>(size_t,size_t,size_t,const unsigned short*,const float*,unsigned short*,bool,bool);
template void gemm<unsigned short,unsigned short,float>(size_t,size_t,size_t,const unsigned short*,const unsigned short*,float*,bool,bool);
template void gemm<float,unsigned short,float>(size_t,size_t,size_t,const float*,const unsigned short*,float*,bool,bool);
template void gemm<unsigned short,float,float>(size_t,size_t,size_t,const unsigned short*,const float*,float*,bool,bool);
template void gemm<float,float,unsigned short>(size_t,size_t,size_t,const float*,const float*,unsigned short*,bool,bool);
template void gemm<float,float,float>(size_t,size_t,size_t,const float*,const float*,float*,bool,bool);

// ============================================================
// Thread control
// ============================================================
void gemm_fp16_set_num_threads(int n) {
#ifdef _OPENMP
    omp_set_num_threads(n);
#endif
    (void)n;
}
int gemm_fp16_get_num_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
