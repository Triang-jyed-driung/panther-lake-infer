/*
 * rwkv_tune.h — runtime tuning: CPU affinity + OMP spin-wait
 *
 * Call rwkv_tune_once() on the inference thread before running the engine.
 * All OpenMP workers share CPUs 0..N-1, matching one multi-core OMP place.
 * No OMP_PLACES, OMP_PROC_BIND, or taskset command is required.
 *
 * Compile flags:
 *   -DRWKV_BIND_CORES=N   use N threads on cores 0..N-1 (default: 4)
 *   -DRWKV_SPIN_WAIT      enable OMP spin-wait (ACTIVE + long blocktime)
 */
#pragma once

#include <cstdlib>
#include <cstdio>
#include <omp.h>

#ifndef RWKV_BIND_CORES
#define RWKV_BIND_CORES 4
#endif

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

static inline void rwkv_tune_once(int n_cores = RWKV_BIND_CORES)
{
    if (n_cores <= 0) {
        fprintf(stderr, "[rwkv_tune] invalid core count: %d\n", n_cores);
        return;
    }
#ifdef __linux__
    if (n_cores > CPU_SETSIZE) {
        fprintf(stderr, "[rwkv_tune] core count exceeds CPU_SETSIZE\n");
        return;
    }
#endif

    omp_set_dynamic(0);
    omp_set_num_threads(n_cores);

    /* OpenMP can initialize binding before main(), so setting environment
     * variables here is too late. Set the actual affinity of every worker,
     * including the caller, after the runtime has created the team. */
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < n_cores; i++) CPU_SET(i, &set);
    int failures = 0;
    int team_size = 0;
    #pragma omp parallel reduction(+:failures)
    {
        #pragma omp single
        team_size = omp_get_num_threads();
        if (sched_setaffinity(0, sizeof(set), &set) != 0)
            failures++;
    }
    if (failures)
        fprintf(stderr, "[rwkv_tune] WARNING: affinity failed for %d threads\n", failures);
    else
        fprintf(stderr, "[rwkv_tune] %d threads share cores 0-%d\n", team_size, n_cores - 1);
    if (team_size != n_cores)
        fprintf(stderr, "[rwkv_tune] WARNING: requested %d threads, runtime supplied %d\n",
                n_cores, team_size);
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
