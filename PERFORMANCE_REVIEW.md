# Performance review — 2026-09-26

The strongest measured opportunity is to route single-row prefill projections through the existing SIMD GEMV kernel. A temporary prototype reduced 256-token CPU prefill from a median 326.97 ms to 206.08 ms (1.59× throughput, 37% less latency). Production source files were not modified.

## Measurements

Hardware: Intel Core Ultra 5 325; CPU affinity 0–3; four OpenMP threads unless stated otherwise. Model: `spec_decode_aligned.bin`, FP16, 12 layers, dimension 768, FFN dimension 3072, vocabulary 65536. No governor or system configuration changes were made. NPU recommendations below are code-review findings, not measured NPU speedups.

The experiment freshly compiled a temporary source copy with the benchmark's optimization flags: `-O3 -march=native -fopenmp -flto -funroll-loops -falign-loops=64`. Each measurement included a warmup. Three paired prefill measurements each averaged three 256-token runs with state reset. Input tokens were 0–255, not a natural-language workload.

| Experiment | Original | Prototype / alternative |
|---|---:|---:|
| 256-token prefill, trial 1 | 327.47 ms | 203.35 ms |
| 256-token prefill, trial 2 | 326.97 ms | 206.43 ms |
| 256-token prefill, trial 3 | 326.59 ms | 206.08 ms |
| Head projection, median of 3 trials | 116.72 ms via GEMM | 1.39 ms via existing GEMV |
| Transposed-head kernel, median | 1.39 ms existing GEMV | 1.61 ms transposed |

The head microbenchmark used actual model weights and a synthetic input vector; its timing is not a whole-model decode timing. Maximum absolute prefill logit difference between original and GEMV prototype: 8.77e-5. This is a limited numerical check, not full FP16/BF16 correctness certification.

Decode thread sweep, 100 tokens per setting, all restricted to cores 0–3:

| Threads | ms/token |
|---:|---:|
| 1 | 9.13 |
| 2 | 5.98 |
| 4 | 4.48 |
| 8 | 7.51 |

These are single thread-sweep samples and demonstrate oversubscription on four cores; they do not compare four cores against all eight physical cores. The existing prebuilt benchmark separately measured 4.74 ms/token and 369.9 ms/256-token prefill, illustrating why paired freshly compiled comparisons matter.

Temporary experiment files and raw output are in `/tmp/rwkv-perf-review/`: `audit.cpp`, `audit`, `results.txt`, `check.cpp`, and `correctness.txt`. They may disappear when temporary storage is cleared.

## Recommended changes, in priority order

### 1. Dispatch single-row projections to GEMV — measured

At `src/rwkv_lib.cpp:688`, `matmul_1()` calls generic GEMM. Its 8-row microkernel cannot handle M=1: `mc_full` becomes zero (`src/gemm_fp16.cpp:561`), and all output values use the scalar remainder loop at line 599. It even packs weights at line 571 that this remainder loop does not use.

Replace the body of `matmul_1()` with `gemv_f16(x, W, y, M, N)`. This covers the final vocabulary projection and the final-layer O/FFN projections. The numbers above measure this exact substitution. Also consider a dedicated short-sequence route for T=2–7, where the generic GEMM similarly misses its full 8-row kernel; the project already has short-window kernels. Such a route must preserve state semantics and avoid computing unused logits/snapshots.

Do not prioritize transposing the decode head: the existing `gemv_f16_head()` was slower in this experiment. The regular head is 96 MiB, so avoiding unnecessary head evaluations is attractive before attempting a new layout or quantization.

### 2. Make application threading match the benchmark — high priority

At review time, the benchmark explicitly selected four threads. Chat also selected four through `setup_cpu()` in `src/npu/common.h`; the API constructor only set caller affinity. These separate setup paths have since been consolidated in `rwkv_tune_once()`, which sets the thread count and applies the same CPU 0–3 mask to every OpenMP worker, including the caller.

More seriously, `api/inference_engine.h` first pins the calling thread to cores 0–3, then `api/main.cpp:59` repins that same thread to core 4. HTTP handlers call `engine_->generate()` synchronously (`api/http_server.h:198` and 258); no inference worker is created. Thus inference runs on the repinned thread, and OpenMP worker placement depends on runtime initialization/environment.

Use one dedicated inference worker, pin it and its OpenMP team to the intended cores, explicitly set the thread count, and keep the HTTP event loop on its own core. Verify actual per-thread affinity under load. The API impact has not been measured; the thread sweep above only supports avoiding oversubscription.

### 3. Avoid unnecessary work and overlap CPU/NPU prefill — unmeasured

`src/npu/spec.cpp:77` finishes all CPU prefill before starting NPU prefill, although the models process the same known tokens independently. Overlap these stages with separate buffers and a final synchronization. Ideal latency changes from their sum toward their maximum, subject to shared-memory contention.

The CPU prefill logits are unused here. Likewise, the all-accepted draft catch-up at `src/npu/spec.cpp:164` computes logits that are discarded before the next decode. Add an explicit state-only path that skips the output head. Passing a null output pointer to the current implementation is not sufficient: it currently writes through it.

For longer prompts, evaluate a separate NPU prefill graph with a larger window, final logits only, and no per-token rollback outputs. The current verification graph exports `[L,n,H,hs,hs]` FP32 recurrent snapshots plus other state outputs. Their WKV payload alone is `4*L*n*H*hs*hs` bytes per window. Those outputs are useful for verification rollback but unnecessary for complete prompt windows. Compiled graph availability, memory usage, transfer cost, and device timing need measurement.

### 4. Remove hot-path allocation and copies — lower-risk follow-up

- `src/gemv_f16.h:242`: two temporary vectors are allocated per sparse FFN call, or 24 allocations per token for this 12-layer model. Reuse capacity in inference-owned scratch buffers.
- `src/npu/spec.cpp:119`: allocate draft snapshots once in the sampler/session and size them from the window, rather than allocating/freeing every speculative round. The current hard-coded three slots are also unsafe for windows greater than four. The final saved draft snapshot is not restored under the current rejection logic and can be eliminated after verifying all stop/rejection cases.
- `src/gemm_fp16.cpp:543` and 620: FP32 GEMM allocates an intermediate full C matrix, writes tiles there, then copies it to the caller. For proven non-aliasing call sites, write completed tiles directly to the caller's output. Investigate reusing packed weights across row tiles rather than packing the same B panels again for each `ic` block; caching all weights as FP32 would increase memory substantially.

Measure each change separately. Allocation counts and redundant passes are visible in the code, but their latency contribution has not been isolated.

### 5. Reduce speculative verification work — unmeasured

`src/npu/spec.cpp:155` builds every target distribution before verification. Build distributions in acceptance order and stop at the first rejection; later rows are unused. Preserve the acceptance/residual distribution math and RNG behavior. Record committed tokens per round, rejection position, draft time, NPU time, sampler time, and rollback time before changing the window size. Larger windows only help if extra accepted tokens outweigh extra drafting and verification cost.

A persistent OpenMP team could also reduce the roughly five parallel-region entries per layer during CPU decode. Profile region/barrier time first; this is a larger restructuring with more synchronization risk than the measured GEMV substitution.

## Correctness issues to resolve before accepting performance results

1. **CPU prefill's last-layer shortcut uses the wrong previous FFN input for T>1.** At `src/rwkv_lib.cpp:1306`, only the last token's O projection and LN2 are computed. Its FFN time shift then uses the state from before the whole chunk, instead of the penultimate token's LN2 output. A 16-token comparison against sequential decode showed max logit error 0.135159. In a temporary copy, disabling this shortcut reduced the error to 3.43e-5 and made the saved states match exactly in that test. Preserve the optimization by computing the necessary penultimate and final O/LN2 rows, using the original state when T=1. Final-layer FFN output is needed only for the final token. The 1.59× prefill result above retains the original shortcut in both variants, so it is not a measurement of the eventual fully corrected implementation.

2. **Partial NPU prompt windows advance through unused slots.** `src/npu/spec.cpp:85` fills only `last_batch` rows but runs the fixed-size graph, whose assignments store the last slot's state. Roll back to `last_batch-1` when the final window is partial, or use a graph that commits the valid length. Reject empty prompt input or define its behavior explicitly.

3. **API initial logits are not preserved correctly.** `api/inference_engine.h:167` and 173 discard the final-row pointer returned by prefill, and generation obtains the beginning of the output tensor instead. Complete cache hits restore recurrent state but the cache format does not restore next-token logits. Save the correct logits with a cached state and use the final valid prompt row. Faster prefix reuse must preserve this invariant.

For the first implementation pass, fix prefill equivalence, apply the measured GEMV dispatch, and correct application thread placement. Validate full logits, saved recurrent states, and continuation after varied prompt lengths/chunk boundaries; then remeasure CPU and end-to-end speculative throughput independently.
