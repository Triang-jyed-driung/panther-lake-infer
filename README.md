
```bash
# 1. Download models (e.g. from HuggingFace)
#    - Draft model: rwkv7-g1d-0.1b-20260129-ctx8192.pth
#    - Target model: rwkv7-g1i-7b-20260129-ctx8192.pth

# 2. Convert CPU weights (draft model)
python scripts/convert_weights.py rwkv7-g1d-0.1b-20260129-ctx8192.pth

# 3. Build NPU window graph (target model)
python scripts/npu/build_window_graph.py rwkv7-g1i-7b-20260129-ctx8192.pth 4

# 4. Compile NPU blob
python scripts/npu/compile_blob.py scripts/npu/export/window_v6_g1i_7b_n4.xml

# 5. Build chat example
make chat

# 6. Run
./build/chat rwkv7_0.1b_fp16.bin scripts/npu/export/window_v6_g1i_7b_n4.blob scripts/npu/export/emb_g1i_7b.f16.bin 1.0 0.9 1.0 0.9
```


# Optimized RWKV-7 inference on Panther Lake x86 CPUs. Supports both fp16 and bf16 weight formats via a single shared library.


## Overview

This project implements fast RWKV-7 inference on x86 CPUs with optional NPU acceleration.

### CPU Inference Engine

The CPU inference engine (`src/`) supports RWKV-7 models of any size in fp16 or bf16 format. It can work standalone for single-model inference.

Model switching and concurrent inference are **not supported**. Each application
engine object is bound to one model for its lifetime. Serialize all inference
calls, even when using separate states or engine objects. CPU scratch buffers
are shared within the process, so a new object alone does not isolate a new
model: use a fresh process and a new engine object when changing models.
Weight files are assumed to be trusted, well-formed output from the bundled converter.

### Speculative Decoding

For faster generation, the project supports speculative decoding with:
- **CPU Draft Model** — Small model (e.g. 0.1B) for fast token proposals
- **NPU Target Model** — Large model (e.g. 7B) via OpenVINO NPU window graph

## Requirements

- CPU: Intel Panther Lake x86-64 with 4 P-cores
- Compiler: Only tested on g++
- Libraries: OpenMP
- For NPU support: OpenVINO 2026.3.1 with NPU plugin

### Weight Conversion

```bash
python scripts/convert_weights.py [input.pth] [output.bin] [--dtype {fp16,bf16}]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `input.pth` | Input .pth checkpoint | `rwkv7-g1d-0.1b-20260129-ctx8192.pth` |
| `output.bin` | Output .bin file | `rwkv7_0.1b_<dtype>.bin` |
| `--dtype` | Output format: `fp16` or `bf16` | `fp16` |

### NPU Window Graph Compilation

For NPU-accelerated speculative decoding, compile the window graph:

```bash
# Build window graph (XML + embedding table)
python scripts/npu/build_window_graph.py [model.pth] [window_size]

# Compile to NPU blob
python scripts/npu/compile_blob.py [model.xml] [device=NPU]
```

| Script | Description |
|--------|-------------|
| `build_window_graph.py` | Build window graph with state (ReadValue/Assign) |
| `compile_blob.py` | Compile XML to NPU blob |

Output files:
- `export/window_v6_*.xml` — OpenVINO graph
- `export/window_v6_*.blob` — Compiled NPU blob
- `export/emb_*.f16.bin` — Embedding table (fp16)

### Build Targets

| Command | Output |
|---------|--------|
| `make all` | Shared library + all tests + chat |
| `make lib` | `librwkv.so` |
| `make test_speed` | Speed benchmark |
| `make test_correct` | Correctness test |
| `make test_chat_callback` | Newline stop-detection regression test |
| `make test_npu_control` | Fixed-window host-control regression (OpenVINO CPU fixture) |
| `make chat` | Interactive chat example |
| `make clean` | Remove build artifacts |

### Project Structure

```
src/
├── rwkv.h              # C API header
├── rwkv_api.cpp        # fp16/bf16 dispatcher
├── rwkv_lib.cpp        # Engine implementation
├── gemm_fp16.cpp/h     # GEMM kernels
├── tokenizer/
│   ├── rwkv_tokenizer.hpp  # Trie tokenizer
│   └── vocab_data.hpp      # Vocabulary data
├── sampling/
│   ├── llm_sampler.h       # LLM sampler (temp + top-p)
│   └── prefix_sum_v2.cpp   # Prefix sum for sampling
├── npu/
│   ├── common.h            # Common utilities
│   ├── npu_window.h        # NPU window wrapper
│   └── spec.cpp            # Speculative decoding implementation
└── chat_callback.h         # Chat callback automaton

examples/
└── chat.cpp            # Interactive chat example

third_party/
└── openvino/           # OpenVINO headers (vendor)
```

## C API

```c
#include "rwkv.h"

// Load model (auto-detects fp16/bf16 from file magic)
RWKVWeights *w = rwkv_weights_load("model.bin");
RWKVState *s = rwkv_state_new(w);

// Query dimensions
int dim = rwkv_get_dim(w);
int vocab = rwkv_get_vocab(w);

// Allocate output buffer
float *logits = malloc(vocab * sizeof(float));

// Single-token decode (scratch buffers managed internally)
rwkv_decode(w, s, token_id, logits);

// Multi-token prefill
int tokens[] = {1, 2, 3, 4, 5};
rwkv_prefill(w, s, tokens, 5, logits);

// Reset state for new sequence
rwkv_state_reset(s);

// Cleanup
rwkv_state_free(s);
rwkv_weights_free(w);
```

### API Reference

| Function | Description |
|----------|-------------|
| `rwkv_weights_load(path)` | Load model (fp16 or bf16, auto-detected) |
| `rwkv_weights_free(w)` | Free model memory |
| `rwkv_state_new(w)` | Allocate new state |
| `rwkv_state_free(s)` | Free state |
| `rwkv_state_reset(s)` | Reset state to zeros |
| `rwkv_decode(w, s, token, logits)` | Single-token forward pass |
| `rwkv_prefill(w, s, tokens, T, logits)` | Multi-token prefill |
| `rwkv_get_dim(w)` | Get embedding dimension |
| `rwkv_get_vocab(w)` | Get vocabulary size |
| `rwkv_get_n_layers(w)` | Get number of layers |
| `rwkv_get_n_heads(w)` | Get number of attention heads |
| `rwkv_get_h_size(w)` | Get head size |
| `rwkv_get_ffn_dim(w)` | Get FFN hidden dimension |
| `rwkv_state_size(w)` | Get state size in floats |
| `rwkv_state_save(s, dst)` | Save state to buffer |
| `rwkv_state_restore(s, src)` | Restore state from buffer |
| `rwkv_spec_prefill(w, s, tokens, T, logits, states)` | Speculative window prefill |
| `rwkv_debug_embed(w, token, out)` | Get token embedding (debug) |

## Chat Example (Speculative Decoding)

The chat example uses speculative decoding with a CPU draft model (0.1B) and NPU target model (7B) for accelerated inference.

### Chat Template

```
User: Hello, how are you?

Assistant: I'm doing well, thank you! How can I help you today?

User: What's the weather like?

Assistant: ...
```

### Usage

```bash
# Build chat example
make chat

# Run with speculative decoding
./build/chat draft.bin blob_path emb.f16.bin

# Run with custom temperature and top-p
./build/chat draft.bin blob_path emb.f16.bin 0.8 0.9 0.7 0.85
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `draft.bin` | Draft model (CPU 0.1B) | required |
| `blob_path` | NPU window graph (7B) | required |
| `emb.f16.bin` | Embedding table (fp16) | required |
| `temp_d` | Draft temperature | 1.0 |
| `top_p_d` | Draft top-p | 1.0 |
| `temp_t` | Target temperature | same as draft |
| `top_p_t` | Target top-p | same as draft |

### Controls

- Type your message and press Enter to get a response
- Type `reset` to start a new conversation
- Type `quit` or `exit` to exit

### Performance Metrics

The chat example displays performance metrics for each response:

```
User: Hello
Assistant:
[prefill: 5.34 tokens/s, 7 tokens]
 Hello! How can I assist you today?

[generate: 11.96 tokens/s, 10 tokens]
```

| Metric | Description |
|--------|-------------|
| `prefill` | Speed of processing input tokens (RNN state update) |
| `generate` | Speed of generating output tokens (speculative decoding) |

Metrics are displayed in gray (ANSI color code `\033[90m`) on separate lines.

### Chat Callback Automaton

The callback implements a state machine for stop condition detection. It stops on `\n\n` (two newlines) or token 0.

| State | Condition | Transition |
|-------|-----------|------------|
| 0 (NORMAL) | Token has `\n\n` | → 2 (STOP) |
| 0 (NORMAL) | Token ends with `\n` | → 1 (LAST_NEWLINE) |
| 0 (NORMAL) | Otherwise | Stay 0 |
| 1 (LAST_NEWLINE) | Token starts with `\n` | → 2 (STOP) |
| 1 (LAST_NEWLINE) | Token has `\n\n` internally | → 2 (STOP) |
| 1 (LAST_NEWLINE) | Otherwise | Same as state 0 |
| 2 (STOP) | — | Stop generation |

### Implementation

- `src/npu/spec_sampler.h` — SpecSampler for sampling and verification
- `src/npu/spec.cpp` — Core speculative decoding functions
- `src/chat_callback.h` — Chat callback automaton for stop conditions
- `examples/chat.cpp` — Interactive chat with speculative decoding

## Test Environment & Performance

### Hardware
- **CPU**: Intel Core Ultra 5 325 (Panther Lake)
- **CPU Mode**: performance governor (max frequency)
- **Cores**: 4 P-cores pinned via `taskset -c 0-3`

### 0.1B Model (rwkv7_0.1b_fp16.bin)

| Metric | Result |
|--------|--------|
| Decode speed | 240 tok/s (4.2 ms/tok) |
| Prefill speed (256 tokens) | 1152 tok/s (222 ms) |
| Correctness (10-step argmax) | 10/10 PASS |

### Speculative Window (Reference Only)

Note: Speculative window does not provide actual speedup for inference. Results below are for implementation reference only.

| Window | Window Speed | Sequential Speed | Speedup |
|--------|--------------|------------------|---------|
| n=4 | 418 tok/s (9.56 ms) | 227 tok/s (17.63 ms) | 1.84x |
| n=6 | 533 tok/s (11.25 ms) | 228 tok/s (26.34 ms) | 2.34x |
| n=8 | 587 tok/s (13.64 ms) | 230 tok/s (34.82 ms) | 2.55x |

See [CPU optimization notes](CPU_OPTIMIZATION.md) for the current kernel changes,
validation commands, and before/after measurements.

### Performance Tuning

```bash
# Set CPU to performance mode (requires root)
pkexec bash -c 'echo "performance" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'
pkexec bash -c 'for i in {0..7}; do cat /sys/devices/system/cpu/cpu$i/cpufreq/scaling_max_freq > /sys/devices/system/cpu/cpu$i/cpufreq/scaling_min_freq; done'

# Run with P-core binding
./build/test_speed model.bin
```

The benchmark, chat example, and API server configure four OpenMP threads in
code. Every inference worker shares the CPU 0–3 affinity mask, including the
calling thread; no `OMP_PLACES`, `OMP_PROC_BIND`, or `taskset` is needed.
The API currently performs inference synchronously on its event-loop thread,
so that thread retains the same CPU mask. CPU governor/frequency settings above
remain separate system settings requiring administrator access.

## Notes

- The `.git` directory is locked due to sensitive content. Please backup your work manually.

### GPU target (OpenVINO)

Compile the same window IR for GPU; the separate output preserves the NPU blob:

```bash
python scripts/npu/compile_blob.py scripts/npu/export/window_v6_g1i_7b_n4.xml GPU
make chat BUILD_DIR=build/gpu
cp build/gpu/chat build/chat_gpu
./chat_gpu.sh
```

The launcher uses `g1d_0.1b.bin` and the graph/embedding under
`scripts/npu/export`. Override `RWKV_DRAFT`, `RWKV_GPU_BLOB`,
`RWKV_EMBEDDING`, or `OV_PYTHON` for other paths. It loads the GPU plugin
from the Python OpenVINO installation because the minimal `build/libs`
bundle contains only the NPU plugin. `RWKV_DEVICE=GPU` selects GPU import
and explicit state uploads on rollback; the default remains NPU.
The GPU export uses latency mode and fp16 inference. Pass `--precision f32`
to the exporter to request fp32 inference (substantially higher GPU memory use).

GPU validation on the remote Core Ultra X7 358H / Arc B390 (OpenVINO
2026.3.1): fp16 export completed in 51.10 seconds, producing a
13,959,147,520-byte blob. `test_gpu_control` passed state readback,
rollback replay, reset replay, and finite-logit checks with zero failures.
The same sampling settings as `chat.sh` yielded 12.53 tokens/s for
"Hello" (10 tokens) and 7.47 tokens/s for the sky explanation (87 tokens).
Earlier NPU runs measured 11.89 and 10.06 tokens/s respectively, but the
latter produced 826 tokens, so the longer runs are not an identical-output
comparison. The fp32 GPU compilation failed under memory pressure; fp16
is the tested configuration. Logs on the remote machine are in
`build/gpu_export_fp16.log`, `build/gpu_control.log`, `build/gpu_chat.log`,
and `build/gpu_chat_long.log`.
