/*
 * test_embed.cpp — Compare embedding for token 12605
 * ===================================================
 * Builds with scalar kernels to isolate the issue.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

#include <immintrin.h>

#include "rwkv.h"
#include "rwkv_tune.h"
#include "gemm_fp16.h"

static inline __m256 load_f16x8(const uint16_t *p) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "rwkv7_7b_fp16.bin";
    RWKVWeights *W = rwkv_weights_load(path);
    if (!W) { fprintf(stderr, "Failed\n"); return 1; }

    int dim   = rwkv_get_dim(W);
    int vocab = rwkv_get_vocab(W);

    /* Access the emb pointer through the opaque struct.
     * The Weights struct layout is known: dim is first, data16 follows.
     * But we can't easily access emb without knowing the struct.
     * Instead, do a decode with a tiny buffer and capture the first x. */

    RWKVState *s = rwkv_state_new(W);
    int buf_sz = 0;
    float *buf = (float *)aligned_alloc(64, buf_sz * sizeof(float));
    float *logits = (float *)aligned_alloc(64, vocab * sizeof(float));

    /* The scratch buffer starts with x = token embedding.
     * After decode, buf[0..dim-1] will have been overwritten.
     * So we need to catch it during the first layer.
     * Alternative: we know the struct layout:
     *   struct Weights { int dim; int n_heads; ... uint16_t *data16; const uint16_t *emb; ... }
     * Let's use pointer arithmetic to find emb. */

    /* Offset of emb in struct (in pointer-sized units):
     * dim(4) n_heads(4) h_size(4) n_layers(4) vocab(4) ffn_dim(4) = 24 bytes
     * data16(8) emb(8) head(8) ln_out_w(8) ln_out_b(8) ... 
     * On 64-bit: after 6 ints (24 bytes) + 1 pointer (8) = 32 bytes to emb pointer.
     * But alignment means the pointers start at offset 32.
     * emb is at offset: 24 (ints) + 8 (data16) = 32
     */
    uintptr_t base = (uintptr_t)W;
    /* Skip: dim(4) n_heads(4) h_size(4) n_layers(4) vocab(4) ffn_dim(4) = 24 
     * data16 ptr(8) = 32
     * emb ptr is at offset 32 */
    const uint16_t *emb = *(const uint16_t **)(base + 32);

    int token = 12605;
    float x[4096];
    for (int i = 0; i < dim; i += 8)
        _mm256_storeu_ps(x + i, load_f16x8(emb + token * dim + i));

    printf("Embedding for token %d (first 8):\n", token);
    for (int i = 0; i < 8; i++) printf("  [%d] = %.6f\n", i, x[i]);
    printf("  mean = %.6f\n", [&]{ double s=0; for(int i=0;i<dim;i++)s+=x[i]; return (float)(s/dim); }());

    free(logits); free(buf);
    rwkv_state_free(s);
    rwkv_weights_free(W);
    return 0;
}
