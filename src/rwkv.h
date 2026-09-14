/*
 * rwkv.h — RWKV-7 CPU Inference Library (Public API)
 * ===================================================
 *
 * Provides:
 *   - Weights loading from fp16 binary format
 *   - State management for recurrent hidden states
 *   - forward_one() — single-token decode
 *   - forward_seq() — multi-token prefill
 *
 * All model dimensions are read from the .bin header at load time,
 * so a single binary works with any RWKV-7 model size.
 *
 * Build as shared library:
 *   g++ -O3 -march=native -fopenmp -shared -fPIC -o librwkv.so src/rwkv_lib.cpp src/gemm_fp16.cpp -lm
 *   g++ -O3 -march=native -fopenmp -shared -o rwkv.dll src/rwkv_lib.cpp src/gemm_fp16.cpp -lm    (Windows)
 */


#ifndef RWKV_H_
#define RWKV_H_


#include <cstdint>
#include <cstdio>

/* ── DLL export / import ─────────────────────────────────────────── */
#if defined(_WIN32) || defined(_WIN64)
    #ifdef RWKV_BUILD_DLL
        #define RWKV_API __declspec(dllexport)
    #else
        #define RWKV_API __declspec(dllimport)
    #endif
#else
    #define RWKV_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── Opaque types ─────────────────────────────────────────────────── */
typedef struct RWKVWeights RWKVWeights;
typedef struct RWKVState   RWKVState;

/* ── Lifecycle ────────────────────────────────────────────────────── */

/* Load weights from a .bin file (fp16 format). Returns NULL on failure. */
RWKV_API RWKVWeights *rwkv_weights_load(const char *path);

/* Free weights. Safe to call with NULL. */
RWKV_API void rwkv_weights_free(RWKVWeights *w);

/* ── Model dimensions (query after load) ──────────────────────────── */
RWKV_API int rwkv_get_dim     (RWKVWeights *w);
RWKV_API int rwkv_get_n_heads (RWKVWeights *w);
RWKV_API int rwkv_get_h_size  (RWKVWeights *w);
RWKV_API int rwkv_get_n_layers(RWKVWeights *w);
RWKV_API int rwkv_get_vocab   (RWKVWeights *w);
RWKV_API int rwkv_get_ffn_dim (RWKVWeights *w);

/* ── State ────────────────────────────────────────────────────────── */

/* Allocate and zero-initialise a new state sized for the given model. */
RWKV_API RWKVState *rwkv_state_new(RWKVWeights *w);

/* Free state. Safe to call with NULL. */
RWKV_API void rwkv_state_free(RWKVState *s);

/* Reset state to zeros (e.g. for a new conversation). */
RWKV_API void rwkv_state_reset(RWKVState *s);

/* ── Inference ────────────────────────────────────────────────────── */

/*
 * Single-token decode.
 *
 *   w      — loaded weights
 *   s      — persistent state (updated in-place)
 *   token  — input token id (0 .. VOCAB-1)
 *   logits — output buffer [VOCAB] floats (written by this function)
 *
 * Internal scratch buffers are managed automatically (static).
 */
RWKV_API void rwkv_decode(
    RWKVWeights *w, RWKVState *s,
    int token, float *logits);

/*
 * Multi-token prefill (sequence processing).
 *
 *   w      — loaded weights
 *   s      — persistent state (updated in-place)
 *   tokens — array of T token ids
 *   T      — number of tokens (≥ 1)
 *   logits — output buffer [VOCAB] (only logits for the LAST token are
 *            computed — sufficient for next-token prediction)
 *
 * Internal scratch buffers are managed automatically (static).
 */
RWKV_API void rwkv_prefill(
    RWKVWeights *w, RWKVState *s,
    const int *tokens, int T,
    float *logits);

/* ── Speculative decoding ──────────────────────────────────────────── */

/* Size of ONE state snapshot in floats:
 *   (2*DIM + N_HEADS * H_SIZE²) * N_LAYERS  (= (H_SIZE+2)*DIM*N_LAYERS
 *   when N_HEADS*H_SIZE == DIM, e.g. (64+2)*layers*dim for H_SIZE=64).
 * The caller pre-allocates all snapshot buffers. */
RWKV_API int rwkv_state_size(RWKVWeights *w);

/* Copy the current recurrent state into dst
 * (dst pre-allocated with rwkv_state_size(w) floats). */
RWKV_API void rwkv_state_save(RWKVState *s, float *dst);

/* Restore the state from a snapshot written by rwkv_state_save() or
 * rwkv_spec_prefill() — used to roll back after accepting a prefix. */
RWKV_API void rwkv_state_restore(RWKVState *s, const float *src);

/*
 * Windowed verification prefill for speculative decoding.
 *
 *   tokens      — T token ids (window of n = 4/6/8 recommended; any T,
 *                 longer inputs are processed in chunks of 8)
 *   logits      — [T * VOCAB] logits AFTER EACH token (logits + i*VOCAB
 *                 is the distribution following token i)
 *   states      — [T * state_size] state snapshot AFTER EACH token:
 *                 snapshot i lives at states + i*rwkv_state_size(w) and
 *                 can be passed to rwkv_state_restore() to roll back.
 *                 May be NULL to skip snapshots.
 *
 * Internal scratch buffers are managed automatically (static).
 *
 * On return the persistent state s holds the state after the LAST token
 * (identical to snapshot T-1). Snapshots and logits make the call
 * equivalent to T sequential rwkv_decode() calls, but the batched-GEMV
 * window streams each weight matrix once instead of T times.
 */
RWKV_API void rwkv_spec_prefill(
    RWKVWeights *w, RWKVState *s,
    const int *tokens, int T,
    float *logits, float *states);

/* ── Debug ─────────────────────────────────────────────────────────── */

/* Copy the embedding for the given token into out[DIM] (fp32). */
RWKV_API void rwkv_debug_embed(RWKVWeights *w, int token, float *out);

#ifdef __cplusplus
}
#endif

#endif /* RWKV_H_ */
