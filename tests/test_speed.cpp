/*
 * test_speed.cpp — RWKV-7 Decode & Prefill Speed Benchmarks
 * ==========================================================
 *
 * Build:
 *   g++ -O3 -march=native -fopenmp -flto -funroll-loops \
 *       -falign-loops=64 -o test_speed \
 *       test_speed.cpp ../src/rwkv_lib.cpp ../src/gemm_fp16.cpp \
 *       -I../src -lm -flto
 *
 * Run (P-core affinity and thread count are configured in code):
 *   ./test_speed [model.bin] [decode_iters] [prefill_iters]
 *
 * Iteration counts auto-scale with model size:
 *   Small (dim <= 1024):  decode 200, prefill 10
 *   Large (dim >  1024):  decode  20, prefill  3
 * Override with command-line: ./test_speed model.bin 50 5
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <chrono>

#include <omp.h>

#include "rwkv.h"
#include "rwkv_tune.h"
#include "gemm_fp16.h"

int main(int argc, char **argv)
{
    rwkv_tune_once();
    setbuf(stdout, NULL);

    printf("RWKV-7 Speed Benchmark  (%d OpenMP threads)\n", omp_get_max_threads());

    /* load weights */
    const char *path = argc > 1 ? argv[1] : "rwkv7_0.1b_fp16.bin";
    RWKVWeights *W = rwkv_weights_load(path);
    if (!W) { fprintf(stderr, "Failed to load %s\n", path); return 1; }

    int dim   = rwkv_get_dim(W);
    int vocab = rwkv_get_vocab(W);

    /* auto-scale iterations: fewer for large models */
    int dec_iters = argc > 2 ? atoi(argv[2]) : (dim <= 1024 ? 200 : 20);
    int pre_iters = argc > 3 ? atoi(argv[3]) : (dim <= 1024 ? 10  :  3);

    /* allocate */
    RWKVState *s = rwkv_state_new(W);
    float *logits = (float *)aligned_alloc(64, vocab * sizeof(float));
    if (!logits) { fprintf(stderr, "OOM\n"); return 1; }

    /* warmup */
    for (int i = 0; i < 3; i++)
        rwkv_decode(W, s, 0, logits);

    /* ── decode benchmark ─────────────────────────────────────── */
    {
        rwkv_state_reset(s);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < dec_iters; i++)
            rwkv_decode(W, s, i % 1000, logits);
        auto t1  = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("Decode: %d iters  %.1f tok/s  (%.2f ms/tok)\n",
               dec_iters, dec_iters * 1000.0 / ms, ms / dec_iters);
    }

    /* ── prefill benchmark (256 tokens) ────────────────────────── */
    {
        std::vector<int> tokens(256);
        for (int i = 0; i < 256; i++) tokens[i] = i % 1000;

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < pre_iters; i++) {
            rwkv_state_reset(s);
            rwkv_prefill(W, s, tokens.data(), 256, logits);
        }
        auto t1  = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("Prefill (256): %d iters  %.1f tok/s  (%.1f ms)\n",
               pre_iters, 256.0 * pre_iters * 1000.0 / ms, ms / pre_iters);
    }

    free(logits);
    rwkv_state_free(s);
    rwkv_weights_free(W);
    return 0;
}
