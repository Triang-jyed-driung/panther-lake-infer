/*
 * test_correctness.cpp — RWKV-7 Correctness (autoregressive decode)
 * ==================================================================
 *
 * Runs 10 steps of autoregressive decode starting from a seed token.
 * Compares each step's logits against reference files.
 *
 * Token feeding strategy:
 *   - If reference files exist at each step, reads the ref's argmax as next token.
 *     This keeps the C++ engine in lockstep with the reference even if our
 *     numerical argmax differs by 1 ULP.
 *   - Falls back to C++ argmax if no reference file is present.
 *
 * Reference files: /tmp/rwkv_expected_t{0..9}.bin
 *   Generate with: scripts/gen_ref_7b.py  (for 7B, uses PIP API)
 *   Or:             scripts/save_expected.py (for 0.1B, numpy reference)
 *
 * Build: make test_correct
 * Run:   ./test_correctness [model.bin]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

#include <omp.h>

#include "rwkv.h"
#include "rwkv_tune.h"
#include "gemm_fp16.h"

static int argmax(const float *logits, int vocab)
{
    int best = 0;
    for (int i = 1; i < vocab; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

static void print_logit_head(const float *logits, int vocab)
{
    printf("  logits[0..7]: ");
    for (int i = 0; i < 8; i++) printf("%.3f ", logits[i]);
    printf("\n  Top-5: ");
    std::vector<std::pair<float, int>> top(vocab);
    for (int i = 0; i < vocab; i++) top[i] = { logits[i], i };
    std::partial_sort(top.begin(), top.begin() + 5, top.end(),
        [](auto &a, auto &b) { return a.first > b.first; });
    for (int i = 0; i < 5; i++)
        printf("%d:%.3f ", top[i].second, top[i].first);
    printf("\n");
}

/* Read reference from file. Returns ref's argmax, or -1 if file missing. */
static int read_ref(const char *ref_path, float *expected, int vocab)
{
    FILE *ef = fopen(ref_path, "rb");
    if (!ef) return -1;
    size_t n = fread(expected, sizeof(float), vocab, ef);
    fclose(ef);
    if (n != (size_t)vocab) return -1;
    return argmax(expected, vocab);
}

static int compare_against_ref(const float *our_logits, const float *expected,
                               double *out_max_diff, double *out_corr,
                               int our_argmax, int ref_argmax, int vocab)
{
    double max_diff = 0, mx = 0, my = 0;
    for (int i = 0; i < vocab; i++) { mx += our_logits[i]; my += expected[i]; }
    mx /= vocab; my /= vocab;
    double cov = 0, vx = 0, vy = 0;
    for (int i = 0; i < vocab; i++) {
        double d = fabs((double)our_logits[i] - (double)expected[i]);
        if (d > max_diff) max_diff = d;
        double dx = our_logits[i] - mx, dy = expected[i] - my;
        cov += dx * dy; vx += dx * dx; vy += dy * dy;
    }
    double corr = cov / sqrt(vx * vy);
    *out_max_diff = max_diff; *out_corr = corr;

    printf("  vs ref: max_diff=%.4f  corr=%.6f  argmax=%d (ref=%d) %s\n",
           max_diff, corr, our_argmax, ref_argmax,
           our_argmax == ref_argmax ? "MATCH" : "MISMATCH");
    return (our_argmax == ref_argmax) ? 0 : 1;
}

int main(int argc, char **argv)
{
    rwkv_tune_once();
    setbuf(stdout, NULL);
    omp_set_num_threads(4);
    gemm_fp16_set_num_threads(4);

    printf("RWKV-7 Correctness — autoregressive decode (10 steps)\n\n");

    const char *path = argc > 1 ? argv[1] : "rwkv7_0.1b_fp16.bin";
    RWKVWeights *W = rwkv_weights_load(path);
    if (!W) { fprintf(stderr, "Failed to load %s\n", path); return 1; }

    int vocab = rwkv_get_vocab(W);

    RWKVState *s = rwkv_state_new(W);
    float *logits   = (float *)aligned_alloc(64, vocab * sizeof(float));
    float *expected = (float *)aligned_alloc(64, vocab * sizeof(float));
    int    buf_sz   = 0;
    float *buf      = (float *)aligned_alloc(64, buf_sz * sizeof(float));
    if (!logits || !expected || !buf) { fprintf(stderr, "OOM\n"); return 1; }

    rwkv_state_reset(s);
    int token = 12605;  // seed token
    int failures = 0;

    for (int step = 0; step < 10; step++) {
        rwkv_decode(W, s, token, logits);
        int our_argmax = argmax(logits, vocab);
        printf("Step %d: token_in=%d  argmax=%d\n", step, token, our_argmax);
        print_logit_head(logits, vocab);

        char ref_path[128];
        snprintf(ref_path, sizeof(ref_path), "/tmp/rwkv_expected_t%d.bin", step);
        int ref_argmax = read_ref(ref_path, expected, vocab);

        if (ref_argmax >= 0) {
            double max_diff, corr;
            int r = compare_against_ref(logits, expected, &max_diff, &corr,
                                        our_argmax, ref_argmax, vocab);
            if (r > 0) failures++;
            // Follow the reference token sequence to stay in lockstep
            token = ref_argmax;
        } else {
            printf("  [SKIP] ref not found: %s\n\n", ref_path);
            // Fall back to our own argmax
            token = our_argmax;
        }
        printf("\n");
    }

    free(buf);
    free(expected);
    free(logits);
    rwkv_state_free(s);
    rwkv_weights_free(W);

    printf("Done.  failures=%d\n", failures);
    return failures > 0 ? 1 : 0;
}
