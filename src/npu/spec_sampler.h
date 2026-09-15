// spec_sampler.h — 投机解码采样器封装
#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <functional>

#include "../rwkv.h"
#include "../sampling/llm_sampler.h"
#include "common.h"

using namespace llm_sampling;

// ===========================================================================
// SpecSampler — 采样与验证封装
// ===========================================================================

struct SpecSampler {
    int V, npose, D, n;  // D=模型维度, n=窗口大小
    float temperature, top_p;        // 草稿
    float temperature_t, top_p_t;    // 目标
    uint64_t rng;

    int*  wP;        // [npose*V] 草稿权重行
    int*  wQ;        // [npose*V] 目标权重行
    int*  pose;      // [npose]   提议 token
    int*  wtmp;      // [V]       临时权重行
    float* probs;    // [V]       softmax 暂存
    float* row32;    // [V]       f16→f32 暂存
    int*  prefix;    // [V]       前缀和
    int*  residual;  // [V]       残差
    int*  block_sums;
    float* emb_in;   // [n*D]     嵌入输入缓冲区
    float* d_logits; // [V]       草稿 logits 缓冲区

    SpecSampler(int V_, int nprop_, int D_, int n_,
                float td, float pd, float tt, float pt, uint64_t rng_)
        : V(V_), npose(nprop_), D(D_), n(n_), temperature(td), top_p(pd),
          temperature_t(tt), top_p_t(pt), rng(rng_) {
        wP         = (int*)  aligned_alloc(64, (size_t)npose * V * 4);
        wQ         = (int*)  aligned_alloc(64, (size_t)npose * V * 4);
        pose       = (int*)  aligned_alloc(64, (size_t)npose * 4);
        wtmp       = (int*)  aligned_alloc(64, (size_t)V * 4);
        probs      = (float*)aligned_alloc(64, (size_t)V * 4);
        row32      = (float*)aligned_alloc(64, (size_t)V * 4);
        prefix     = (int*)  aligned_alloc(64, (size_t)V * 4);
        residual   = (int*)  aligned_alloc(64, (size_t)V * 4);
        block_sums = (int*)  aligned_alloc(64, (size_t)(V / 2048 + 1) * 4);
        emb_in     = (float*)aligned_alloc(64, (size_t)n * D * sizeof(float));
        d_logits   = (float*)aligned_alloc(64, (size_t)V * sizeof(float));
    }
    ~SpecSampler() {
        free(wP); free(wQ); free(pose); free(wtmp); free(probs);
        free(row32); free(prefix); free(residual); free(block_sums);
        free(emb_in); free(d_logits);
    }

    // 从 f32 logits 采样
    int sample(const float* logits, float temp, float tp) {
        int total = build_row_weights(logits, V, temp, tp,
                                      probs, wtmp, prefix, block_sums);
        return (int)categorical_sample(prefix, V, total, next_rand(rng));
    }

    // 从 f16 logits 采样
    int sample_f16(const uint16_t* lg16, float temp, float tp) {
        for (int j = 0; j < V; j++) row32[j] = f16_to_f32(lg16[j]);
        return sample(row32, temp, tp);
    }

    // 草稿提议：构建权重行 + 采样
    int propose(int i, const float* d_logits) {
        int total = build_row_weights(d_logits, V, temperature, top_p,
                                      probs, wP + (size_t)i * V, prefix, block_sums);
        return (int)categorical_sample(prefix, V, total, next_rand(rng));
    }

    // 构建目标权重行
    void build_target(int row, const uint16_t* lg16) {
        for (int j = 0; j < V; j++) row32[j] = f16_to_f32(lg16[j]);
        build_row_weights(row32, V, temperature_t, top_p_t,
                          probs, wQ + (size_t)(row - 1) * V, prefix, block_sums);
    }

    // 投机验证
    std::tuple<int, int, bool> verify(int T, std::function<bool(int)> callback) {
        auto [fail_pos, correction] = speculative_verify(wP, wQ, pose, npose, V, residual, prefix, block_sums, rng);
        int v=T, done_pos=0;
        for(;;) {
            if (callback(v)) return {done_pos, -1, true};
            // 完成则返回包含本位置，验证失败不包含本位置，且完成优先于验证失败，故>=
            if (done_pos >= fail_pos) return {fail_pos, correction, false};
            v = pose[done_pos++];
        }
    }
};
