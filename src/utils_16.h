/*
 * utils_16.h — unified 16-bit float ↔ fp32 conversion API
 * ==========================================================
 *
 * Single home for every 16-bit ↔ fp32 conversion used across the
 * codebase (weight loading, GEMV, GEMM, embedding).  All definitions
 * are `static inline` right here in the header so they inline into
 * the hot SIMD kernels in every translation unit — no reliance on
 * -flto / LTO.  utils_16.cpp is retained only as an empty TU for
 * builds whose Makefiles still list it.
 *
 *   fp16  (IEEE 754 half, 1+5+10)  — hardware F16C conversion
 *   bf16  (bfloat16,     1+8+7)    — bit-shift conversion; exact
 *                                    for finite values (bf16 ⊂ fp32)
 *
 * NOTE: instead of #pragma once this header uses a classic
 * #ifndef/#define guard, because src/rwkv_api.cpp includes it TWICE
 * in one translation unit — once inside `namespace Fp16` (without
 * RWKV_WEIGHTS_BF16) and once inside `namespace Bf16` (with it) —
 * #undef-ing the guard macro between the two namespace blocks so
 * each namespace compiles its own format-specialized copy.
 */

#ifndef UTILS_16_H_
#define UTILS_16_H_

#include <cstdint>
#include <cstddef>
#include <immintrin.h>

/* ── fp16 → fp32 ─────────────────────────────────────────────────── */
static inline float fp16_to_f32(uint16_t h)
{
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16((short)h)));
}

/* Load 8 fp16 → 8 fp32 (one 128-bit load + F16C) */
static inline __m256 load_fp16x8(const uint16_t *p)
{
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
}

/* ── fp32 → fp16 (round-to-nearest-even) ─────────────────────────── */
static inline uint16_t f32_to_fp16(float v)
{
    return (uint16_t)_mm_extract_epi16(
        _mm_cvtps_ph(_mm_set_ss(v), _MM_FROUND_TO_NEAREST_INT), 0);
}

/* Store 8 fp32 → 8 fp16 (RNE) */
static inline void store_f32x8_fp16(void *p, __m256 v)
{
    _mm_storeu_si128((__m128i *)p,
        _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
}

/* ── bf16 → fp32 ─────────────────────────────────────────────────── */
/* bf16 is the top 16 bits of an fp32 value: zero-extend u16→u32
 * (_mm256_cvtepu16_epi32) then shift left 16 (_mm256_slli_epi32)
 * places the bf16 bits into the fp32 layout exactly. */
static inline float bf16_to_f32(uint16_t b)
{
    __m256i z = _mm256_cvtepu16_epi32(_mm_set1_epi16((short)b));
    return _mm256_cvtss_f32(_mm256_castsi256_ps(_mm256_slli_epi32(z, 16)));
}

/* Load 8 bf16 → 8 fp32 (cvtepu16_epi32 + slli_epi32, exact) */
static inline __m256 load_bf16x8(const uint16_t *p)
{
    __m256i z = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p));
    return _mm256_castsi256_ps(_mm256_slli_epi32(z, 16));
}

/* ── fp32 → bf16 (round-to-nearest-even) ─────────────────────────── */
static inline uint16_t f32_to_bf16(float v)
{
    /* RNE at the 16-bit truncation point — pure SSE integer ops */
    __m128i u    = _mm_castps_si128(_mm_set_ss(v));
    __m128i lsb  = _mm_and_si128(_mm_srli_epi32(u, 16), _mm_set1_epi32(1));
    __m128i bias = _mm_add_epi32(_mm_set1_epi32(0x7fffu), lsb);
    __m128i res  = _mm_srli_epi32(_mm_add_epi32(u, bias), 16);
    return (uint16_t)_mm_cvtsi128_si32(res);
}

/* Store 8 fp32 → 8 bf16 (RNE) */
static inline void store_f32x8_bf16(void *p, __m256 v)
{
    __m256i u   = _mm256_castps_si256(v);
    __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16),
                                   _mm256_set1_epi32(1u));
    u = _mm256_add_epi32(u, _mm256_add_epi32(_mm256_set1_epi32(0x7fffu), lsb));
    __m256i hi = _mm256_srli_epi32(u, 16);        /* values ≤ 0xFFFF */

    __m128i lo    = _mm256_castsi256_si128(hi);
    __m128i hi128 = _mm256_extracti128_si256(hi, 1);
    __m128i packed = _mm_packus_epi32(lo, hi128); /* clamp = identity here */
    _mm_storeu_si128((__m128i *)p, packed);
}

/* ── weight storage format — selected at COMPILE TIME ────────────── */
/* The engine's weight binaries store 16-bit floats as either fp16
 * (magic "RWKV7") or bf16 (magic "RWKV7B").  Compile with
 * -DRWKV_WEIGHTS_BF16 to decode bf16 weight binaries.  The
 * format-agnostic helpers below resolve to the selected format with
 * zero runtime branching. */
#ifdef RWKV_WEIGHTS_BF16
static inline float  to_f32(uint16_t x)           { return bf16_to_f32(x); }
static inline __m256 load_16x8(const uint16_t *p) { return load_bf16x8(p); }
#else
static inline float  to_f32(uint16_t x)           { return fp16_to_f32(x); }
static inline __m256 load_16x8(const uint16_t *p) { return load_fp16x8(p); }
#endif

/* ── bulk conversion ─────────────────────────────────────────────── */
/* Convert n 16-bit floats to fp32 using the compile-time-selected
 * format (macro above) — no runtime branching inside the loop. */
static inline void convert_16_to_f32(const uint16_t *src, float *dst, size_t n)
{
    size_t i = 0;
#ifdef RWKV_WEIGHTS_BF16
    /* bf16: exact shift-based conversion, 8 lanes/iter */
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i, load_bf16x8(src + i));
    for (; i < n; i++)
        dst[i] = bf16_to_f32(src[i]);
#else
    /* fp16: hardware F16C conversion, 8 lanes/iter */
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i, load_fp16x8(src + i));
    for (; i < n; i++)
        dst[i] = fp16_to_f32(src[i]);
#endif
}

#endif /* UTILS_16_H_ */
