/*
 * test_prefill.cpp — Prefill correctness against PIP reference
 * =============================================================
 * Reads tokens from /tmp/rwkv_prefill_tokens.bin,
 * runs rwkv_prefill, compares against /tmp/rwkv_prefill_ref.bin
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>

#include "rwkv.h"
#include "rwkv_tune.h"
#include "gemm_fp16.h"

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    omp_set_num_threads(4);
    gemm_fp16_set_num_threads(4);

    const char *path = argc > 1 ? argv[1] : "rwkv7_7b_fp16.bin";
    RWKVWeights *W = rwkv_weights_load(path);
    if (!W) { fprintf(stderr, "Failed\n"); return 1; }

    int vocab = rwkv_get_vocab(W);

    // Read tokens
    FILE *tf = fopen("/tmp/rwkv_prefill_tokens.bin", "rb");
    if (!tf) { fprintf(stderr, "No tokens file\n"); return 1; }
    uint32_t T; fread(&T, 4, 1, tf);
    std::vector<int> tokens(T);
    for (uint32_t i = 0; i < T; i++) { int32_t t; fread(&t, 4, 1, tf); tokens[i] = t; }
    fclose(tf);

    printf("Prefill %u tokens: ", T);
    for (uint32_t i = 0; i < T && i < 20; i++) printf("%d ", tokens[i]);
    printf("\n");

    // Load reference
    float *expected = (float *)aligned_alloc(64, vocab * sizeof(float));
    FILE *ef = fopen("/tmp/rwkv_prefill_ref.bin", "rb");
    if (!ef) { fprintf(stderr, "No ref\n"); return 1; }
    fread(expected, sizeof(float), vocab, ef);
    fclose(ef);

    // Run prefill
    RWKVState *s = rwkv_state_new(W);
    float *logits = (float *)aligned_alloc(64, vocab * sizeof(float));
    int buf_sz = 0;
    float *buf = (float *)aligned_alloc(64, buf_sz * sizeof(float));

    rwkv_prefill(W, s, tokens.data(), T, logits);

    // Compare
    double maxd = 0, mx = 0, my = 0;
    for (int i = 0; i < vocab; i++) { mx += logits[i]; my += expected[i]; }
    mx /= vocab; my /= vocab;
    double cov = 0, vx = 0, vy = 0;
    int our_top = 0, ref_top = 0;
    for (int i = 0; i < vocab; i++) {
        double d = fabs((double)logits[i] - (double)expected[i]);
        if (d > maxd) maxd = d;
        double dx = logits[i]-mx, dy = expected[i]-my;
        cov += dx*dy; vx += dx*dx; vy += dy*dy;
        if (logits[i] > logits[our_top]) our_top = i;
        if (expected[i] > expected[ref_top]) ref_top = i;
    }
    double corr = cov / sqrt(vx*vy);

    printf("logits[0..7]: ");
    for (int i = 0; i < 8; i++) printf("%.3f ", logits[i]);
    printf("\n");
    printf("ref  [0..7]: ");
    for (int i = 0; i < 8; i++) printf("%.3f ", expected[i]);
    printf("\n");
    printf("max_diff=%.4f  corr=%.6f  argmax=%d (ref=%d) %s\n",
           maxd, corr, our_top, ref_top,
           our_top == ref_top ? "MATCH" : "MISMATCH");

    free(buf); free(logits); free(expected);
    rwkv_state_free(s); rwkv_weights_free(W);
    return our_top == ref_top ? 0 : 1;
}
