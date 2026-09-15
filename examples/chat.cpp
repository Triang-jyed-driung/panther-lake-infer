/*
 * chat.cpp — RWKV-7 命令行聊天示例（投机解码）
 * ================================================
 *
 * 使用CPU草稿模型(0.1B) + NPU窗口图(7B)进行投机解码聊天
 *
 * 用法：
 *   ./chat <draft.bin> <blob> <emb.f16.bin> [temp_d] [top_p_d] [temp_t] [top_p_t]
 *
 * 聊天模板：
 *   User: something?\n\n
 *   Assistant: Answer\n\n
 *   User: another round\n\n
 *   Assistant: ...
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <memory>
#include <stdexcept>

#include "rwkv.h"
#include "tokenizer/rwkv_tokenizer.hpp"
#include "sampling/llm_sampler.h"
#include "chat_callback.h"
#include "npu/npu_window.h"
#include "npu/common.h"
#include "npu/spec_sampler.h"
#include "npu/spec_api.h"

using namespace llm_sampling;
using namespace rwkv;

// 聊天机器人类
class ChatBot {
public:
    ChatBot(const char* draft_path, const char* blob_path, const char* emb_path,
            float temperature = 1.0f, float top_p = 1.0f,
            float temperature_t = 1.0f, float top_p_t = 1.0f)
        : rng_(20260906) {

        // 加载草稿模型
        draft_ = rwkv_weights_load(draft_path);
        if (!draft_)
            throw std::runtime_error(std::string("Failed to load draft: ") + draft_path);
        printf("Draft: dim=%d vocab=%d\n", rwkv_get_dim(draft_), rwkv_get_vocab(draft_));

        // 加载NPU模型
        npu_.load(blob_path);

        // 加载嵌入表
        emb_ = EmbeddingTable::load(emb_path, npu_.D);
        if (!emb_)
            throw std::runtime_error("Failed to load embedding table");

        // 创建状态
        state_ = rwkv_state_new(draft_);

        // 创建投机采样器（管理emb_in和d_logits缓冲区）
        sampler_.reset(new SpecSampler(npu_.V, npu_.n - 1, npu_.D, npu_.n,
                                      temperature, top_p, temperature_t, top_p_t, rng_));

        // 创建回调
        callback_.reset(new ChatCallback(&tokenizer_));

        printf("Ready. Window size = %d\n\n", npu_.n);
    }

    ~ChatBot() = default;

    // 重置对话
    void reset_conversation() {
        npu_.reset_state();
        rwkv_state_reset(state_);
    }

    // 获取回复
    std::string get_response(const std::string& user_input) {
        // RWKV是RNN，状态已编码历史，只需prefill新增token
        std::string new_text = "User: " + user_input + "\n\nAssistant:";

        // 编码新增token
        std::vector<int32_t> tokens = tokenizer_.encode(new_text);

        // 使用spec.cpp中的prefill函数（只处理新增token，状态保留历史）
        // 返回最后位置的 logits 指针
        double t_prefill_start = now_ms();
        const uint16_t* lg = prefill(npu_, draft_, state_, emb_, *sampler_, tokens);
        double t_prefill_end = now_ms();
        double prefill_ms = t_prefill_end - t_prefill_start;
        printf("\n\033[90m[prefill: %.2f tokens/s, %d tokens]\033[0m\n", tokens.size() * 1000.0 / prefill_ms, (int)tokens.size());

        // 准备生成
        callback_->reset();
        int generated = 0;
        const int max_tokens = 2048;
        int next_token = -1;  // -1 表示由 spec_decoder 从 logits 采样

        double t0 = now_ms();

        // 投机解码循环
        while (generated < max_tokens) {
            DecodeResult res = spec_decoder(npu_, draft_, state_, emb_, *sampler_,
                                            lg, next_token,
                                            [this](int id) { return callback_->on_token(id); });

            for (int tok : res.committed) {
                printf("%s", tokenizer_.decode({tok}).c_str());
                fflush(stdout);
                generated++;
            }

            if (res.next < 0) break;
            // spec_decoder 已更新 NPU 状态，重新获取 logits
            lg = static_cast<const uint16_t*>(
                npu_.req.get_output_tensor(0).data());
            next_token = res.next;
        }

        double t1 = now_ms();
        if (generated > 0)
            printf("\n\033[90m[generate: %.2f tokens/s, %d tokens]\033[0m\n", generated * 1000.0 / (t1 - t0), generated);
        return callback_->current_text();
    }

private:
    NpuWindow npu_;
    RWKVWeights* draft_;
    RWKVState* state_;
    EmbeddingTable emb_;
    TrieTokenizer tokenizer_;
    std::unique_ptr<SpecSampler> sampler_;
    std::unique_ptr<ChatCallback> callback_;
    uint64_t rng_;
};

int main(int argc, char** argv) {
    setbuf(stdout, NULL);

    if (argc < 4) {
        fprintf(stderr,
            "用法: %s <draft.bin> <blob> <emb.f16.bin> [temp_d] [top_p_d] [temp_t] [top_p_t]\n"
            "\n"
            "投机解码聊天示例\n"
            "  draft.bin    - 草稿模型 (CPU 0.1B)\n"
            "  blob         - NPU窗口图 (7B)\n"
            "  emb.f16.bin  - 嵌入表 (fp16)\n"
            "  temp_d       - 草稿温度 (默认1.0)\n"
            "  top_p_d      - 草稿top_p (默认1.0)\n"
            "  temp_t       - 目标温度 (默认与草稿相同)\n"
            "  top_p_t      - 目标top_p (默认与草稿相同)\n"
            "\n"
            "输入 'quit' 或 'exit' 退出\n"
            "输入 'reset' 重置对话\n",
            argv[0]);
        return 1;
    }

    const char* draft_path = argv[1];
    const char* blob_path  = argv[2];
    const char* emb_path   = argv[3];
    float temperature   = argc > 4 ? (float)atof(argv[4]) : 1.0f;
    float top_p         = argc > 5 ? (float)atof(argv[5]) : 1.0f;
    float temperature_t = argc > 6 ? (float)atof(argv[6]) : temperature;
    float top_p_t       = argc > 7 ? (float)atof(argv[7]) : top_p;

    printf("RWKV-7 Speculative Chat\n");
    printf("=======================\n");
    printf("Draft: %s\n", draft_path);
    printf("NPU: %s\n", blob_path);
    printf("Embedding: %s\n", emb_path);
    printf("Temperature: %.2f / %.2f (draft/target)\n", temperature, temperature_t);
    printf("Top-p: %.2f / %.2f (draft/target)\n", top_p, top_p_t);
    printf("=======================\n\n");

    setup_cpu();

    try {
        ChatBot bot(draft_path, blob_path, emb_path, temperature, top_p, temperature_t, top_p_t);

        std::string line;
        while (true) {
            printf("\nUser: ");
            if (!std::getline(std::cin, line)) break;

            // 去除首尾空白
            size_t start = line.find_first_not_of(" \t\n\r");
            if (start == std::string::npos) continue;
            size_t end = line.find_last_not_of(" \t\n\r");
            line = line.substr(start, end - start + 1);

            if (line.empty()) continue;
            if (line == "quit" || line == "exit") {
                printf("Goodbye!\n");
                break;
            }
            if (line == "reset") {
                bot.reset_conversation();
                printf("对话已重置。\n");
                continue;
            }

            printf("\nAssistant:");
            bot.get_response(line);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
