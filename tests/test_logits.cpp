// test_logits.cpp — 对比 CPU decode 的 logits
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include "rwkv.h"
#include "rwkv_tune.h"

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    if (argc < 2) { fprintf(stderr, "用法: %s <model.bin> [tok0 tok1 ...]\n", argv[0]); return 1; }

    RWKVWeights* W = rwkv_weights_load(argv[1]);
    if (!W) { fprintf(stderr, "load failed\n"); return 1; }
    int dim = rwkv_get_dim(W), vocab = rwkv_get_vocab(W);
    printf("CPU model: dim=%d vocab=%d\n", dim, vocab);

    RWKVState* s = rwkv_state_new(W);
    float* logits = (float*)aligned_alloc(64, vocab * 4);

    // 从命令行读取 token
    std::vector<int> toks;
    for (int i = 2; i < argc; i++) toks.push_back(atoi(argv[i]));
    if (toks.empty()) toks = {10464, 11685};

    rwkv_state_reset(s);
    printf("tokens:");
    for (int t : toks) printf(" %d", t);
    printf("\n\n");

    for (size_t i = 0; i < toks.size(); i++) {
        rwkv_decode(W, s, toks[i], logits);
        // top-5
        std::vector<size_t> idx(vocab);
        for (int j = 0; j < vocab; j++) idx[j] = j;
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                          [&](size_t a, size_t b) { return logits[a] > logits[b]; });
        printf("after tok[%zu]=%d  top5:", i, toks[i]);
        for (int k = 0; k < 5; k++) printf(" %zu(%.3f)", idx[k], logits[idx[k]]);
        printf("\n");
    }

    free(logits);
    rwkv_state_free(s);
    rwkv_weights_free(W);
    return 0;
}
