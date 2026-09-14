/*
 * test_spec.cpp — Speculative-decode window (rwkv_spec_prefill) test
 * ==================================================================
 *
 * Validates the windowed verification path against sequential decode:
 *
 *   1. Logits equivalence: for window sizes n = 4/6/8, the logits after
 *      each token from rwkv_spec_prefill() must match the logits of the
 *      corresponding rwkv_decode() step.
 *   2. State snapshots: the snapshot after each token must match the
 *      sequential-decode state (rwkv_state_save after each step).
 *   3. Rollback: restoring snapshot[m] and decoding the next token must
 *      reproduce the sequential result from that point.
 *   4. Speed: windowed tok/s vs sequential decode tok/s for n = 4/6/8.
 *
 * Build: make test_spec
 * Run:   ./build/test_spec [model.bin]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>

#include <omp.h>

#include "rwkv.h"
#include "rwkv_tune.h"

using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

/* deterministic pseudo-random token stream (lcg) */
static int next_token(int &seed, int vocab)
{
    seed = (int)((1103515245u * (unsigned)seed + 12345u) & 0x7fffffffu);
    return seed % vocab;
}

static double max_abs_diff(const float *a, const float *b, size_t n)
{
    double m = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char **argv)
{
    rwkv_tune_once();
    setbuf(stdout, NULL);
    omp_set_num_threads(4);

    printf("RWKV-7 Speculative-window test (n = 4/6/8)\n\n");

    const char *path = argc > 1 ? argv[1] : "rwkv7_0.1b_fp16.bin";
    RWKVWeights *W = rwkv_weights_load(path);
    if (!W) { fprintf(stderr, "Failed to load %s\n", path); return 1; }

    int dim      = rwkv_get_dim(W);
    int vocab    = rwkv_get_vocab(W);
    int s_sz     = rwkv_state_size(W);
    printf("Model: dim=%d vocab=%d  state_size=%d floats (%.2f MB/snapshot)\n\n",
           dim, vocab, s_sz, s_sz * 4.0 / 1024 / 1024);

    RWKVState *s = rwkv_state_new(W);

    float *dec_buf = (float *)aligned_alloc(64, 0 * 4);
    // float *sp_buf  = (float *)aligned_alloc(64, 0 * 4);
    float *lg_seq  = (float *)aligned_alloc(64, 8 * vocab * 4);   /* [8][vocab] */
    float *lg_win  = (float *)aligned_alloc(64, 8 * vocab * 4);
    float *st_seq  = (float *)aligned_alloc(64, 8 * (size_t)s_sz * 4);
    float *st_win  = (float *)aligned_alloc(64, 8 * (size_t)s_sz * 4);
    float *lg_tmp  = (float *)aligned_alloc(64, vocab * 4);
    float *st_tmp  = (float *)aligned_alloc(64, (size_t)s_sz * 4);
    if (!dec_buf || !lg_seq || !lg_win || !st_seq || !st_win ||
        !lg_tmp || !st_tmp) { fprintf(stderr, "OOM\n"); return 1; }

    int failures = 0;

    for (int n : {4, 6, 8}) {
        printf("════ window n = %d ════\n", n);

        /* deterministic token window */
        std::vector<int> toks(n + 1);
        int seed = 42 + n;
        for (int t = 0; t < n + 1; t++) toks[t] = next_token(seed, vocab);

        /* ── reference: sequential decode + snapshot after each step ── */
        rwkv_state_reset(s);
        for (int t = 0; t < n; t++) {
            rwkv_decode(W, s, toks[t], lg_seq + (size_t)t * vocab);
            rwkv_state_save(s, st_tmp);
            memcpy(st_seq + (size_t)t * s_sz, st_tmp, (size_t)s_sz * 4);
        }
        /* reference continuation: decode token n from state after n-1 */
        rwkv_decode(W, s, toks[n], lg_tmp);

        /* ── windowed speculative prefill ── */
        rwkv_state_reset(s);
        rwkv_spec_prefill(W, s, toks.data(), n, lg_win, st_win);

        /* ── 1. logits equivalence ── */
        for (int t = 0; t < n; t++) {
            double d = max_abs_diff(lg_seq + (size_t)t * vocab,
                                    lg_win + (size_t)t * vocab, vocab);
            bool ok = d < 1e-3;
            printf("  logits[%d]: max_diff=%.2e  %s\n", t, d, ok ? "OK" : "FAIL");
            if (!ok) failures++;
        }

        /* ── 2. state snapshots equivalence ── */
        for (int t = 0; t < n; t++) {
            double d = max_abs_diff(st_seq + (size_t)t * s_sz,
                                    st_win + (size_t)t * s_sz, s_sz);
            bool ok = d < 1e-4;
            printf("  state[%d]:  max_diff=%.2e  %s\n", t, d, ok ? "OK" : "FAIL");
            if (!ok) failures++;
        }

        /* ── 3. rollback: restore snapshot[n-1], decode toks[n] ── */
        rwkv_state_restore(s, st_win + (size_t)(n - 1) * s_sz);
        rwkv_decode(W, s, toks[n], lg_win);
        {
            double d = max_abs_diff(lg_tmp, lg_win, vocab);
            bool ok = d < 1e-3;
            printf("  rollback decode after snapshot[%d]: max_diff=%.2e  %s\n",
                   n - 1, d, ok ? "OK" : "FAIL");
            if (!ok) failures++;
        }

        /* ── 4. speed: window vs sequential ────────────────────── */
        const int reps = 30;
        rwkv_state_reset(s);
        auto t0 = Clock::now();
        for (int r = 0; r < reps; r++)
            rwkv_spec_prefill(W, s, toks.data(), n, lg_win, st_win);
        auto t1 = Clock::now();
        double win_ms = elapsed_ms(t0, t1) / reps;

        rwkv_state_reset(s);
        t0 = Clock::now();
        for (int r = 0; r < reps; r++)
            for (int t = 0; t < n; t++)
                rwkv_decode(W, s, toks[t], lg_seq);
        t1 = Clock::now();
        double seq_ms = elapsed_ms(t0, t1) / reps;

        printf("  speed: window %.2f ms (%.0f tok/s)  vs  %dx decode %.2f ms (%.0f tok/s)  → %.2fx\n",
               win_ms, n * 1000.0 / win_ms,
               n, seq_ms, n * 1000.0 / seq_ms,
               seq_ms / win_ms);
        printf("\n");
    }

    free(dec_buf); /* free(sp_buf); */ free(lg_seq); free(lg_win);
    free(st_seq); free(st_win); free(lg_tmp); free(st_tmp);
    rwkv_state_free(s);
    rwkv_weights_free(W);

    printf("Done.  failures=%d\n", failures);
    return failures > 0 ? 1 : 0;
}
