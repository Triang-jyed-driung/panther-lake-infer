// spec_api.h — 投机解码API声明
// 供chat.cpp等外部程序使用
#pragma once

#include <vector>
#include <functional>
#include <cstdint>

#include "npu_window.h"
#include "spec_sampler.h"
#include "../rwkv.h"
#include "common.h"

// 投机解码结果
struct DecodeResult {
    std::vector<int> committed;  // 本轮提交的 token
    int next;                    // 下一个词元，-1 表示 done
};

// prefill — 阅读模式
// 同步处理 token 序列，两模型状态一致推进
// 返回：最后位置的 logits 指针（供 spec_decoder 首次采样）
const uint16_t* prefill(NpuWindow& npu, RWKVWeights* Wd, RWKVState* Dst,
                        EmbeddingTable& emb, SpecSampler& sampler,
                        const int* tokens, int T);

// 便捷重载：接受 vector（int32_t 版本）
inline const uint16_t* prefill(NpuWindow& npu, RWKVWeights* Wd, RWKVState* Dst,
                               EmbeddingTable& emb, SpecSampler& sampler,
                               const std::vector<int32_t>& tokens) {
    return prefill(npu, Wd, Dst, emb, sampler, tokens.data(), (int)tokens.size());
}

// spec_decoder — 解码模式（一轮投机解码）
// last_token >= 0: 外部指定（上一轮的 next）
// last_token < 0:  从 last_logits 采样
// callback:        每提交一个 token 调用，返回 true 则停止
DecodeResult spec_decoder(NpuWindow& npu, RWKVWeights* Wd, RWKVState* Dst,
                          EmbeddingTable& emb, SpecSampler& sampler,
                          const uint16_t* last_logits, int last_token,
                          std::function<bool(int)> callback);
