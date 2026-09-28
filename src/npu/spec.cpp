// spec.cpp — 投机解码核心模块
// =========================================================================
// 提供两个核心函数：
//   1. prefill()       — 阅读模式，同步处理 token 序列
//   2. spec_decoder()  — 解码模式，一轮投机解码
//
// 状态约定：每轮开始时，大小模型的内部状态都恰好处理完上下文 C。
//
// prefill：同步处理 token 序列，两模型状态一致推进，无采样。
//
// spec_decoder 一轮推理过程：
//   第一步，如果 last_token >= 0，使用 T=last_token，否则从 last_logits 采样 T。
//   第二步，草稿模型读取 T。保存快照 s0 = s(C+T)。草稿输出为 l1。
//   第三步，草稿采样 d1=sample(l1)、d2=sample(l2)、d3=sample(l3)。
//     - d1 喂入草稿，保存快照 s1=s(C+T+d1)
//     - d2 喂入草稿，保存快照 s2=s(C+T+d1+d2)
//     - d3 只采样、暂不喂入
//   第四步，NPU 窗口推理 [T, d1, d2, d3]，得到 4 行 logits + 4 列状态快照。
//   第五步，投机验证：草稿分布 (l1,l2,l3) vs 目标分布 (L1,L2,L3)。
//
// C    T    d1   d2   d3
// |----|>S0-|>S1-|>S2-|>S3
//      L1   L2   L3   L4
//
// 验证结果：
//   1. callback 在 da 位置返回 true（第一种情况）：
//      提交 [T, d1, ..., da]，返回 {committed, -1}
//
//   2. 拒绝 d(a+1)（第二种情况）：
//      提交 [T, d1, ..., da]，回退状态到 Sa=s(C+T+d1+...+da)
//      返回 {committed, R}，R 为改正 token
//
//   3. 全接受：
//      提交 [T, d1, d2, d3]，草稿读 d3 追到 s3
//      从 L4 采样 T'，返回 {committed, T'} 或 {committed, -1}（callback触发）
//
// 每轮结束后，两模型状态对齐（都在 s(C+T+d1+...+da) 或 s3）。
// next（R 或 T'）作为下一轮的 last_token 输入。
//
// 两个模型的状态回退都必须通过 CPU 端的 memcpy 完成。

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <functional>
#include <stdexcept>

#include "../rwkv.h"
#include "../gemm_fp16.h"
#include <omp.h>
#include "../sampling/llm_sampler.h"
#include "npu_window.h"
#include "spec_sampler.h"
#include "spec_api.h"
using namespace llm_sampling;


// DecodeResult已在spec_api.h中定义


// ===========================================================================
// prefill — 阅读模式
// ===========================================================================
// CPU 和 NPU 分别独立处理，最后状态对齐。
// 返回：最后位置的 logits 指针（供 spec_decoder 首次采样）

const uint16_t* prefill(NpuWindow& npu, RWKVWeights* Wd, RWKVState* Dst,
                        EmbeddingTable& emb, SpecSampler& sampler,
                        const int* tokens, int T) {
    const int D = npu.D, n = npu.n;
    if (T <= 0)
        throw std::invalid_argument("prefill requires at least one token");
    const int cpu_win = 256;
    float* emb_in = sampler.emb_in;
    float* d_logits = sampler.d_logits;

    // CPU: 分批处理，每批 cpu_win 个 token
    for (int i = 0; i < T; i += cpu_win)
        rwkv_prefill(Wd, Dst, tokens + i, std::min(cpu_win, T - i), d_logits);

    // NPU: 分窗口处理
    ov::Tensor emb_t(ov::element::f32, ov::Shape{(size_t)n, (size_t)D}, emb_in);
    npu.req.set_input_tensor(0, emb_t);

    int last_batch = 0;
    for (int i = 0; i < T; i += n) {
        last_batch = std::min(n, T - i);
        for (int j = 0; j < last_batch; j++)
            emb.fill_row(emb_in + (size_t)j * D, tokens[i + j]);
        // The graph always runs n slots; unused rows must be initialized.
        if (last_batch < n)
            std::fill(emb_in + (size_t)last_batch * D, emb_in + (size_t)n * D, 0.0f);
        npu.req.infer();
        // Graph Assign nodes commit slot n-1. Keep only the real prompt tokens.
        if (last_batch < n)
            npu.rollback_to(last_batch - 1);
    }

    // 返回最后位置的 logits 指针
    const uint16_t* lg = static_cast<const uint16_t*>(
        npu.req.get_output_tensor(0).data());
    return lg + (last_batch - 1) * (size_t)npu.V;
}


// ===========================================================================
// spec_decoder — 解码模式（一轮投机解码）
// ===========================================================================
// last_token >= 0: 外部指定（上一轮的 next）
// last_token < 0:  从 last_logits 采样
// callback:        每提交一个 token 调用，返回 true 则停止

DecodeResult spec_decoder(NpuWindow& npu, RWKVWeights* Wd, RWKVState* Dst,
                          EmbeddingTable& emb, SpecSampler& sampler,
                          const uint16_t* last_logits, int last_token,
                          std::function<bool(int)> callback) {
    DecodeResult result{{}, -1};
    const int V = npu.V, D = npu.D, n = npu.n;
    const int npose = n - 1;
    if (n < 2)
        throw std::invalid_argument("speculative decoding requires a window of at least two tokens");
    const int dsz = rwkv_state_size(Wd);
    float* emb_in = sampler.emb_in;
    float* d_logits = sampler.d_logits;

    // s0 plus n-2 decoded proposals: n-1 snapshots for any fixed graph size.
    std::vector<float> snapshots((size_t)npose * dsz);
    float* d_snap = snapshots.data();

    // 设置 NPU 输入
    ov::Tensor emb_t(ov::element::f32, ov::Shape{(size_t)n, (size_t)D}, emb_in);
    npu.req.set_input_tensor(0, emb_t);

    // ── 步骤 1：确定 T ──
    int T;
    if (last_token >= 0) {
        T = last_token;
    } else {
        T = sampler.sample_f16(last_logits, sampler.temperature_t, sampler.top_p_t);
    }

    // ── 步骤 2：草稿读 T → 提议 d1,d2,d3 ──
    rwkv_decode(Wd, Dst, T, d_logits);
    rwkv_state_save(Dst, d_snap);  // s0 = s(C+T)

    for (int i = 0; i < npose; i++) {
        sampler.pose[i] = sampler.propose(i, d_logits);
        if (i < npose - 1) {
            rwkv_decode(Wd, Dst, sampler.pose[i], d_logits);
            rwkv_state_save(Dst, d_snap + (size_t)(i + 1) * dsz);
        }
    }

    // ── 步骤 3：NPU 窗口推理 ──
    emb.fill_row(emb_in, T);
    for (int i = 0; i < npose; i++)
        emb.fill_row(emb_in + (size_t)(i + 1) * D, sampler.pose[i]);
    npu.req.infer();

    const uint16_t* lg = static_cast<const uint16_t*>(
        npu.req.get_output_tensor(0).data());

    // 构建目标分布
    for (int a = 0; a < npose; a++)
        sampler.build_target(a + 1, lg + (size_t)a * V);

    // ── 步骤 4：投机验证 ──
    auto [rej, rtok, done] = sampler.verify(T, callback);
    result.committed.push_back(T);
    for (int i = 0; i < rej; i++) 
        result.committed.push_back(sampler.pose[i]);
    if (rej == npose)
        rwkv_decode(Wd, Dst, sampler.pose[npose - 1], d_logits);
        // 此时：草稿状态=s3，NPU状态=S3，对齐
    else {
        npu.rollback_to(rej);
        if (rej < npose - 1)
            rwkv_state_restore(Dst, d_snap + (size_t)rej * dsz);
        // 应当对齐
    }
    if (!done) { // 未完待续
        if (rej == npose) result.next = sampler.sample_f16(lg + npose * (size_t)V, sampler.temperature_t, sampler.top_p_t);
        else result.next = rtok;
    }
    return result;
}


// spec_decoder函数已定义，可被chat.cpp等外部程序调用
