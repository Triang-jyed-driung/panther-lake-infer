# RWKV-7 CPU Inference Engine

Optimized RWKV-7 inference on Panther Lake x86 CPUs. Supports both fp16 and bf16 weight formats via a single shared library.

## Build & Run

### Requirements

- CPU: Intel Panther Lake x86-64 with 4 P-cores
- Compiler: Only tested on g++
- Libraries: OpenMP

### Quick Start

```bash
# Convert weights (.pth → .bin)
python scripts/convert_weights.py model.pth model.bin

# Build everything
make all

# Run speed benchmark
./build/test_speed model.bin

# Run correctness test
./build/test_correctness model.bin
```

### Build Targets

| Command | Output |
|---------|--------|
| `make all` | Shared library + all tests |
| `make lib` | `librwkv.so` |
| `make test_speed` | Speed benchmark |
| `make test_correct` | Correctness test |
| `make clean` | Remove build artifacts |

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

### Performance Tuning

```bash
# Set CPU to performance mode (requires root)
pkexec bash -c 'echo "performance" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'
pkexec bash -c 'for i in {0..7}; do cat /sys/devices/system/cpu/cpu$i/cpufreq/scaling_max_freq > /sys/devices/system/cpu/cpu$i/cpufreq/scaling_min_freq; done'

# Run with P-core binding
OMP_PLACES="{0:4}:1:4" OMP_PROC_BIND=close taskset -c 0-3 ./build/test_speed model.bin
```
