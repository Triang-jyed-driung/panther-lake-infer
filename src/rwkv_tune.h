/*
 * rwkv_tune.h — runtime tuning: CPU affinity + OMP spin-wait
 *
 * Call rwkv_tune_once() at the start of main().
 *
 * Compile flags:
 *   -DRWKV_BIND_CORES=N   pin to cores 0..N-1
 *   -DRWKV_SPIN_WAIT      enable OMP spin-wait (ACTIVE + long blocktime)
 */
#pragma once

#include <cstdlib>
#include <cstdio>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

static inline void rwkv_tune_once()
{
    /* ── CPU affinity ─────────────────────────────────────────── */
#ifdef RWKV_BIND_CORES
#ifdef __linux__
    {
        cpu_set_t set;
        CPU_ZERO(&set);
        for (int i = 0; i < RWKV_BIND_CORES; i++) CPU_SET(i, &set);
        if (sched_setaffinity(0, sizeof(set), &set) == 0)
            fprintf(stderr, "[rwkv_tune] pinned to cores 0-%d\n", RWKV_BIND_CORES - 1);
    }
#endif
#endif

    /* ── OpenMP spin-wait ─────────────────────────────────────── */
#ifdef RWKV_SPIN_WAIT
#ifdef __linux__
    setenv("OMP_WAIT_POLICY", "ACTIVE", 0);
    setenv("GOMP_SPINCOUNT",  "1000000", 0);
    setenv("KMP_BLOCKTIME",   "0",       0);
#endif
#endif
}
