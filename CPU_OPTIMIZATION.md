# CPU prefill and decode optimization

The CPU engine retains FP32 accumulation for both FP16 and BF16 weights.
No external BLAS dependency or additional weight copy is required.

## Correctness follow-up (2026-10-21)

- Last-logit CPU prefill processes at most 256 tokens per internal chunk,
  carrying recurrent state across chunks and computing the vocabulary head
  only for the final chunk. Main activation scratch is bounded by
  `4 * 256 * (12 * dim + 2 * ffn_dim)` bytes (80 MiB for dim=4096,
  ffn_dim=16384), excluding GEMM temporaries and model/state storage.
- Prefill layout uses checked `size_t` multiplication/addition, including the
  allocation byte count and alignment rounding. Engine scratch growth checks
  allocation failure before releasing the previous buffer.
- Optimized/scalar CPU key normalization and the local NumPy references use
  `max(sqrt(sum_sq), 1e-12)`, keeping zero-key normalization finite without
  adding epsilon to ordinary nonzero norms.
- `make test_cpu_edge_cases` builds optimized and scalar regressions for
  zero/tiny/ordinary key norms and large/overflowing buffer sizes without huge
  allocations. Prefill equivalence also covers 255/256/257/513-token prompts
  from reset and nonzero states, including continued decode.

## Changes

- Last-logit prefill computes the final layer's FFN projections for the last
  token only. LN2 still evaluates the preceding token and final token to preserve
  time shift and the final FFN recurrent state. Earlier layers and the full-output
  speculative path retain all token outputs.
- The prefill output head uses GEMV instead of a one-row scalar GEMM fallback.
  Windows of at most eight tokens use batched GEMV, with temporary output storage
  to preserve in-place R/K/V projections.
- Decode retains independent OpenMP sections for dimensions up to 1024 (0.1B and
  0.4B shapes). Larger models serialize those sections and use multi-threaded
  GEMV within projections; no nested active teams are required.
- The common FP32 activation × 16-bit weight GEMM uses a 6×16 AVX2 kernel with
  twelve accumulators. Each 256-row weight panel is converted once per column
  tile and reused across all token rows. A is not repeatedly packed. Column tile
  widths balance narrow gate projections and 768-wide layers across the team.
  Transposed and other type combinations retain the general GEMM path.
- Dense GEMV uses four independent accumulators and aligned output partitions.
  Sparse FFN GEMV retains accumulators across eight nonzero rows, including when
  adding directly into an existing residual.
- Fixed the general GEMM kernel's non-multiple-of-four K remainder handling,
  GEMV row reduction with smaller runtime teams, and sparse GEMV scalar tails.

## Validation

`make test_cpu_kernels test_cpu_kernels_bf16` builds standalone tests for all eight
GEMM type combinations, both transpose flags, odd/tile-boundary dimensions,
in-place projections, dense/sparse GEMV, and one/three/four/dynamic thread teams.
Run the resulting binaries without arguments for correctness, or with `bench`
for representative GEMM timings.

`test_prefill_equivalence` compares logits, recurrent state, and continued decode
against sequential decode. It covers single-token, short-window, 9/16/17/256-token,
chunked, and nonzero initial-state prompts. `test_spec` checks every output logit,
state snapshot, and rollback of the full-output path.

## Measurement procedure

Measurements use the local Core Ultra 5 325 and four threads sharing cores 0–3,
configured by `rwkv_tune_once`. Governor settings were not changed. Before/after
runs execute sequentially without competing benchmarks. These are local timing
samples, not a guarantee for other CPUs or thermal conditions. No 0.4B weights
were available for an end-to-end run.

Original source files and the original speed benchmark are backed up under
`build/cpu_baseline` because this project's Git directory is intentionally locked.
Raw final timing logs are under `build/gemm_{before,after}.log` and
`build/speed_{01,7}_{before,after}.log`.

## Results (2026-09-27)

| Model / operation | Before | After | Throughput gain |
|---|---:|---:|---:|
| 0.1B FP16 decode | 4.56 ms/token | 4.35 ms/token | 4.7% |
| 0.1B FP16 prefill, 256 tokens | 226.1 ms | 170.5 ms | 32.6% |
| 7B FP16 decode | 187.37 ms/token | 181.19 ms/token | 3.4% |
| 7B FP16 prefill, 256 tokens | 11,025.2 ms | 8,818.7 ms | 25.0% |

Decode/prefill iteration counts: 300/8 for 0.1B, 30/3 for 7B. Timing excludes
model loading. The speed benchmark performs three decode warmups and resets
state between prefill iterations. Throughput gains use tokens/s or unrounded
available millisecond measurements; displayed timings are rounded.

Standalone mixed GEMM, FP32 × FP16 → FP32, five measured calls after one warmup:

| M × N × K | Before (ms) | After (ms) | After (GFLOP/s) |
|---|---:|---:|---:|
| 256 × 768 × 768 | 0.939 | 0.903 | 334.6 |
| 256 × 3072 × 768 | 3.690 | 3.080 | 392.1 |
| 256 × 4096 × 4096 | 24.334 | 20.486 | 419.3 |
| 256 × 16384 × 4096 | 92.607 | 82.032 | 418.9 |
| 16 × 4096 × 4096 | 2.820 | 2.674 | 200.7 |

FP16 and BF16 standalone kernel tests passed. The 0.1B FP16/BF16 prefill
regressions and full-output speculative tests passed, including continuation,
state snapshots, and rollback. Full-output speculative execution intentionally
retains its original FFN work for every token.

The final 7B FP16 prefill regression also passed all 124 comparisons,
including 256-token prompts from reset and nonzero initial states. Maximum
absolute difference across logits and state was 0.00032001734; the existing
comparison tolerance is `1e-3 + 1e-4 * scale`.
