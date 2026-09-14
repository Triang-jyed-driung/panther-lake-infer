/*
 * test_utils16.cpp — unit tests for unified 16-bit ↔ fp32 conversions
 * =====================================================================
 *
 * Verifies src/utils_16.{h,cpp}:
 *   - fp16 → fp32 / fp32 → fp16 (exhaustive over all 65536 fp16 patterns)
 *   - bf16 → fp32 (exact; bf16 ⊂ fp32) and fp32 → bf16 (RNE)
 *   - SIMD load/store lanes match the scalar conversions
 *   - bulk convert_16_to_f32 matches the scalar loop (compile-time format)
 *   - precision: bf16 keeps small-magnitude values that fp16 underflows
 *     (the "higher precision" expected from the bf16 path)
 *
 * Build & run:
 *   make test_utils16 && ./build/test_utils16
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <random>

#include "utils_16.h"
#include "rwkv_tune.h"

static int failures = 0;
static int checks   = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        checks++;                                                          \
        if (!(cond)) {                                                     \
            printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);       \
            failures++;                                                    \
        }                                                                  \
    } while (0)

/* ── independent fp16 → fp32 reference (double-based reconstruction) ── */
static float ref_f16_f32(uint16_t h)
{
    int sign = (h >> 15) & 1 ? -1 : 1;
    int e    = (h >> 10) & 0x1f;
    int m    = h & 0x3ff;
    double v;
    if      (e == 0)  v = (m == 0) ? 0.0 : ldexp((double)m, -24);
    else if (e == 31) v = (m == 0) ? INFINITY : NAN;
    else              v = ldexp((double)(0x400 | m), e - 25);
    return (float)(sign * v);
}

static bool same_bits(float a, float b)
{
    uint32_t ua, ub;
    memcpy(&ua, &a, 4);
    memcpy(&ub, &b, 4);
    return ua == ub;
}

static bool is_nan_f(float v) { return v != v; }

/* ── exhaustive fp16 ↔ fp32 ──────────────────────────────────────── */
static void test_fp16_exhaustive()
{
    printf("[1] fp16 ↔ fp32 (exhaustive 65536 patterns)\n");
    for (uint32_t h = 0; h < 65536; h++) {
        uint16_t u = (uint16_t)h;
        float f = fp16_to_f32(u);

        if (is_nan_f(ref_f16_f32(u))) {
            /* NaN → NaN; payload is implementation-defined, compare semantically */
            CHECK(is_nan_f(f));
        } else {
            CHECK(same_bits(f, ref_f16_f32(u)));
            /* round trip: fp16 → fp32 → fp16 is identity */
            CHECK(f32_to_fp16(f) == u);
        }
    }
}

/* ── exhaustive bf16 → fp32 ──────────────────────────────────────── */
static void test_bf16_exhaustive()
{
    printf("[2] bf16 → fp32 (exhaustive 65536 patterns, must be exact)\n");
    for (uint32_t b = 0; b < 65536; b++) {
        float f = bf16_to_f32((uint16_t)b);
        /* bf16 is the top 16 bits of fp32: value bits == (b << 16) */
        uint32_t expect = b << 16;
        uint32_t got;
        memcpy(&got, &f, 4);
        CHECK(got == expect);
    }
}

/* ── SIMD loads match scalars ────────────────────────────────────── */
static void test_simd_loads()
{
    printf("[3] load_fp16x8 / load_bf16x8 match scalar conversions\n");
    std::mt19937 rng(12345);
    std::uniform_int_distribution<uint32_t> dist(0, 65535);

    uint16_t v[8];
    for (int iter = 0; iter < 2000; iter++) {
        for (int i = 0; i < 8; i++) v[i] = (uint16_t)dist(rng);

        __m256 f16 = load_fp16x8(v);
        __m256 b16 = load_bf16x8(v);
        float out16[8], outb[8];
        memcpy(out16, &f16, 32);
        memcpy(outb,  &b16, 32);
        for (int i = 0; i < 8; i++) {
            CHECK(same_bits(out16[i], fp16_to_f32(v[i])));
            CHECK(same_bits(outb[i],  bf16_to_f32(v[i])));
        }
    }
}

/* ── SIMD stores match scalars ───────────────────────────────────── */
static void test_simd_stores()
{
    printf("[4] store_f32x8_fp16 / store_f32x8_bf16 match scalar conversions\n");
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    float vals[8];
    for (int iter = 0; iter < 2000; iter++) {
        for (int i = 0; i < 8; i++) vals[i] = dist(rng);

        __m256 v = _mm256_loadu_ps(vals);
        uint16_t h[8], b[8];
        store_f32x8_fp16(h, v);
        store_f32x8_bf16(b, v);
        for (int i = 0; i < 8; i++) {
            CHECK(h[i] == f32_to_fp16(vals[i]));
            CHECK(b[i] == f32_to_bf16(vals[i]));
        }
    }
}

/* ── fp32 → bf16 rounds to nearest (RNE) ─────────────────────────── */
static void test_f32_to_bf16_rounding()
{
    printf("[5] fp32 → bf16 rounding error ≤ 0.5 ulp\n");
    std::mt19937 rng(4242);
    /* spread exponents: normal + tiny (subnormal bf16) + huge */
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::uniform_int_distribution<int> exp_dist(-140, 120);

    for (int iter = 0; iter < 100000; iter++) {
        float v;
        if (iter % 3 == 0)
            v = ldexpf(dist(rng), exp_dist(rng));   /* extreme exponents */
        else
            v = dist(rng);

        float back = bf16_to_f32(f32_to_bf16(v));

        /* half-ulp bound: |err| ≤ 2^(e-9) for normal bf16, 2^-134 for subnormal */
        double bound;
        if (fabsf(v) < 0x1p-126f)      /* bf16 subnormal range */
            bound = 0x1p-134;
        else {
            int e;
            frexpf(fabsf(v), &e);      /* |v| = m·2^e, m ∈ [0.5,1) */
            bound = ldexp(1.0, e - 9); /* bf16 mantissa = 8 bits → 0.5 ulp = 2^(e-9) */
        }
        double err = fabs((double)back - (double)v);
        CHECK(err <= bound * (1.0 + 1e-12));
    }

    /* special values */
    CHECK(f32_to_bf16(INFINITY) == 0x7F80);
    CHECK(f32_to_bf16(-INFINITY) == 0xFF80);
    uint16_t nanb = f32_to_bf16(NAN);
    CHECK(((nanb >> 7) & 0xFF) == 0xFF);   /* exponent all 1s → NaN/Inf */
}

/* ── bulk conversion matches scalar loop ─────────────────────────── */
static void test_bulk()
{
#ifdef RWKV_WEIGHTS_BF16
    printf("[6] convert_16_to_f32 matches scalar to_f32 (bf16 build)\n");
#else
    printf("[6] convert_16_to_f32 matches scalar to_f32 (fp16 build)\n");
#endif
    std::mt19937 rng(2024);
    std::uniform_int_distribution<uint32_t> dist(0, 65535);

    const size_t MAXN = 1000;
    uint16_t src[MAXN];
    float dst[MAXN];

    /* load_16x8 must match scalar to_f32 (compile-time-selected format) */
    for (int iter = 0; iter < 500; iter++) {
        for (int i = 0; i < 8; i++) src[i] = (uint16_t)dist(rng);
        float out[8];
        __m256 v = load_16x8(src);
        memcpy(out, &v, 32);
        for (int i = 0; i < 8; i++)
            CHECK(same_bits(out[i], to_f32(src[i])));
    }

    for (size_t n = 0; n <= 40; n++) {
        for (size_t i = 0; i < n; i++) src[i] = (uint16_t)dist(rng);
        convert_16_to_f32(src, dst, n);
        for (size_t i = 0; i < n; i++)
            CHECK(same_bits(dst[i], to_f32(src[i])));
    }
    /* random lengths up to MAXN */
    for (int iter = 0; iter < 50; iter++) {
        size_t n = dist(rng) % MAXN;
        for (size_t i = 0; i < n; i++) src[i] = (uint16_t)dist(rng);
        convert_16_to_f32(src, dst, n);
        for (size_t i = 0; i < n; i++)
            CHECK(same_bits(dst[i], to_f32(src[i])));
    }
}

/* ── precision demonstration: bf16 preserves what fp16 loses ─────── */
static void test_precision_demo()
{
    printf("[7] precision: bf16 keeps small values that fp16 underflows\n");
    const float vals[] = { 1e-30f, 1e-20f, 1e-10f, 1e-5f, 1e-3f, 1.0f, 1e3f, 1e30f };
    printf("    %-12s %-14s %-14s\n", "value", "fp16 err", "bf16 err");
    for (float v : vals) {
        float hf  = fp16_to_f32(f32_to_fp16(v));
        float bf  = bf16_to_f32(f32_to_bf16(v));
        double eh = fabs((double)hf - v);
        double eb = fabs((double)bf - v);
        printf("    %-12g %-14.3g %-14.3g\n", v, eh, eb);
        /* small values (< fp16 min subnormal 2^-24 ≈ 6e-8) must survive bf16 */
        if (fabsf(v) < 0x1p-24f && v != 0.0f) {
            CHECK(hf == 0.0f);       /* fp16 underflows to zero */
            CHECK(bf != 0.0f);       /* bf16 still represents it */
            CHECK(eb < eh);          /* bf16 is strictly more precise here */
        }
        /* huge values (> fp16 max 65504) must survive bf16 */
        if (fabsf(v) > 65504.0f) {
            CHECK(std::isinf(hf));        /* fp16 overflows to inf */
            CHECK(std::isfinite(bf));     /* bf16 keeps fp32 range */
        }
    }
}

int main()
{
    rwkv_tune_once();
    setbuf(stdout, NULL);
    printf("utils_16 conversion unit tests\n");
    printf("===============================\n\n");

    test_fp16_exhaustive();
    test_bf16_exhaustive();
    test_simd_loads();
    test_simd_stores();
    test_f32_to_bf16_rounding();
    test_bulk();
    test_precision_demo();

    printf("\n-------------------------------\n");
    printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
