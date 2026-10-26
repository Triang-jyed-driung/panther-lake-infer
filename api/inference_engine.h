/*
 * inference_engine.h — 推理引擎封装
 * ===================================
 *
 * 封装RWKV推理逻辑，支持：
 * - 单token解码
 * - 序列prefill
 * - 投机解码
 * - 状态缓存
 */
#pragma once

#include <QObject>
#include <QMutex>
#include <QMutexLocker>
#include <QString>
#include <QByteArray>
#include <QCryptographicHash>
#include <vector>
#include <cstdint>
#include <functional>
#include <string>
#include <memory>
#include <algorithm>
#include <stdexcept>

#include "openai_types.h"
#include "rwkv.h"
#include "tokenizer/rwkv_tokenizer.hpp"
#include "sampling/llm_sampler.h"
#include "chat_callback.h"
#include "npu/npu_window.h"
#include "npu/spec_sampler.h"
#include "npu/spec_api.h"
#include "npu/common.h"
#include "state_cache.h"

class InferenceEngine : public QObject {
    Q_OBJECT

public:
    explicit InferenceEngine(const char* draft_path,
                             const char* blob_path,
                             const char* emb_path,
                             int max_states = 3,
                             QObject *parent = nullptr)
        : QObject(parent), rng_(20260906) {
        // 初始化推理线程和整个OpenMP线程组的亲和性。
        setup_cpu();

        // 加载草稿模型
        draft_ = rwkv_weights_load(draft_path);
        if (!draft_) {
            throw std::runtime_error(
                std::string("Failed to load draft: ") + draft_path);
        }
        printf("Draft loaded: dim=%d vocab=%d\n",
               rwkv_get_dim(draft_), rwkv_get_vocab(draft_));

        // 加载NPU模型
        npu_.load(blob_path);

        // 加载嵌入表
        emb_ = EmbeddingTable::load(emb_path, npu_.D);
        if (!emb_) {
            throw std::runtime_error("Failed to load embedding table");
        }

        // 创建状态
        state_ = rwkv_state_new(draft_);

        // 创建投机采样器
        sampler_.reset(new SpecSampler(
            npu_.V, npu_.n - 1, npu_.D, npu_.n,
            1.0f, 1.0f, 1.0f, 1.0f, rng_));

        // 创建状态缓存
        cache_.reset(new StateCache(max_states));

        // 获取模型信息
        dim_ = rwkv_get_dim(draft_);
        vocab_ = rwkv_get_vocab(draft_);
        n_layers_ = rwkv_get_n_layers(draft_);

        printf("Engine ready. Vocab=%d, Dim=%d, Layers=%d\n",
               vocab_, dim_, n_layers_);
    }

    ~InferenceEngine() {
        shutdown();
    }

    void shutdown() {
        if (state_) { rwkv_state_free(state_); state_ = nullptr; }
        if (draft_) { rwkv_weights_free(draft_); draft_ = nullptr; }
    }

    // 获取分词器
    const rwkv::TrieTokenizer& tokenizer() const { return tokenizer_; }

    // 获取模型名
    QString modelName() const { return "rwkv-7-0.1b"; }

    // 生成回复
    struct GenerateResult {
        QString text;
        int prompt_tokens;
        int completion_tokens;
        QString finish_reason;
    };

    GenerateResult generate(const std::vector<ChatMessage>& messages,
                           double temperature = 1.0,
                           double top_p = 1.0,
                           int max_tokens = 2048,
                           const QString& stop = QString(),
                           std::function<void(const QString&)> stream_callback = nullptr) {
        return generatePrompt(buildPrompt(messages), temperature, top_p,
                              max_tokens, stop, std::move(stream_callback), true);
    }

    GenerateResult generatePrompt(const std::string& prompt,
                                  double temperature = 1.0,
                                  double top_p = 1.0,
                                  int max_tokens = 2048,
                                  const QString& stop = QString(),
                                  std::function<void(const QString&)> stream_callback = nullptr,
                                  bool stop_on_chat_marker = false) {
        QMutexLocker lock(&mutex_);

        std::vector<int32_t> tokens = tokenizer_.encode(prompt);

        // 更新采样器参数
        sampler_->temperature = (float)temperature;
        sampler_->top_p = (float)top_p;
        sampler_->temperature_t = (float)temperature;
        sampler_->top_p_t = (float)top_p;

        // 检查状态缓存 - O(min(n, max_cached))增量hash
        std::vector<int> token_vec(tokens.begin(), tokens.end());
        int prefix_len = 0;
        QByteArray best_hash;
        
        // 只循环到已缓存状态的最大token数量
        int max_check = std::min((int)token_vec.size(), cache_->maxCachedTokens());
        
        QCryptographicHash hash(StateCache::HASH_ALGORITHM);
        for (int i = 0; i < max_check; i++) {
            hash.addData(QByteArray((const char*)&token_vec[i], sizeof(int)));
            QByteArray h = hash.result().toHex();
            
            QString filepath;
            if (cache_->findByHash(h, filepath)) {
                prefix_len = i + 1;
                best_hash = h;
            }
        }
        
        // 重置状态
        rwkv_state_reset(state_);
        npu_.reset_state();
        
        // Own the last valid prompt row: output tensors are reused by inference.
        std::vector<uint16_t> prompt_logits;
        if (prefix_len > 0 &&
            !cache_->loadByHash(best_hash, state_, npu_, prompt_logits)) {
            prefix_len = 0;
            rwkv_state_reset(state_);
            npu_.reset_state();
        }
        if (prefix_len < (int)token_vec.size()) {
            const uint16_t* last_row = prefill(npu_, draft_, state_, emb_,
                *sampler_, token_vec.data() + prefix_len, (int)token_vec.size() - prefix_len);
            prompt_logits.assign(last_row, last_row + npu_.V);
            cache_->save(token_vec, draft_, state_, npu_, prompt_logits);
        }
        if (prompt_logits.empty())
            throw std::invalid_argument("generation requires a non-empty prompt");

        int prompt_tokens = (int)tokens.size();

        // 生成回复
        ChatCallback callback(&tokenizer_);
        QString generated_text;
        int generated = 0;
        bool stopped_by_chat = false;
        bool stopped_by_sequence = false;
        const uint16_t* lg = prompt_logits.data();
        int next_token = -1;

        // UTF-8缓冲区，用于处理分词器可能产生的不完整UTF-8序列
        QByteArray utf8_buffer;

        // 辅助函数：检查UTF-8序列是否完整
        auto isUtf8Complete = [](const QByteArray& data) -> bool {
            if (data.isEmpty()) return true;
            const unsigned char* bytes = (const unsigned char*)data.constData();
            int len = data.size();
            int i = 0;
            while (i < len) {
                unsigned char b = bytes[i];
                int seq_len = 0;
                if (b < 0x80) {
                    seq_len = 1;
                } else if ((b & 0xE0) == 0xC0) {
                    seq_len = 2;
                } else if ((b & 0xF0) == 0xE0) {
                    seq_len = 3;
                } else if ((b & 0xF8) == 0xF0) {
                    seq_len = 4;
                } else {
                    return false; // 无效的UTF-8起始字节
                }
                if (i + seq_len > len) return false; // 不完整的序列
                // 检查后续字节
                for (int j = 1; j < seq_len; j++) {
                    if ((bytes[i + j] & 0xC0) != 0x80) return false;
                }
                i += seq_len;
            }
            return true;
        };

        // 辅助函数：从缓冲区提取完整的UTF-8字符
        auto extractCompleteUtf8 = [](QByteArray& buffer) -> QString {
            if (buffer.isEmpty()) return QString();
            const unsigned char* bytes = (const unsigned char*)buffer.constData();
            int len = buffer.size();
            int last_complete = 0;
            int i = 0;
            while (i < len) {
                unsigned char b = bytes[i];
                int seq_len = 0;
                if (b < 0x80) {
                    seq_len = 1;
                } else if ((b & 0xE0) == 0xC0) {
                    seq_len = 2;
                } else if ((b & 0xF0) == 0xE0) {
                    seq_len = 3;
                } else if ((b & 0xF8) == 0xF0) {
                    seq_len = 4;
                } else {
                    break; // 无效字节
                }
                if (i + seq_len > len) break; // 不完整
                // 检查后续字节
                bool valid = true;
                for (int j = 1; j < seq_len; j++) {
                    if ((bytes[i + j] & 0xC0) != 0x80) {
                        valid = false;
                        break;
                    }
                }
                if (!valid) break;
                i += seq_len;
                last_complete = i;
            }
            if (last_complete == 0) return QString();
            QString result = QString::fromUtf8(buffer.left(last_complete));
            buffer = buffer.mid(last_complete);
            return result;
        };

        while (generated < max_tokens) {
            int remaining = max_tokens - generated;
            DecodeResult res = spec_decoder(
                npu_, draft_, state_, emb_, *sampler_,
                lg, next_token,
                [&callback, &remaining, &stopped_by_chat, stop_on_chat_marker](int id) {
                    // Token 0 is the model's end marker in both APIs; only
                    // the chat endpoint additionally treats blank lines as EOS.
                    stopped_by_chat = stop_on_chat_marker
                        ? callback.on_token(id) : id == 0;
                    --remaining;
                    // The decoder includes the token that stops the callback.
                    // Stop within the batch so committed state and output agree.
                    return stopped_by_chat || remaining == 0;
                });

            for (int tok : res.committed) {
                std::string token_text = tokenizer_.decode({tok});
                
                // 将token文本添加到UTF-8缓冲区
                utf8_buffer.append(QByteArray(token_text.c_str(), token_text.size()));
                
                // 提取完整的UTF-8字符
                QString complete_chars = extractCompleteUtf8(utf8_buffer);
                if (!complete_chars.isEmpty()) {
                    generated_text += complete_chars;
                    if (stream_callback) {
                        stream_callback(complete_chars);
                    }
                }

                generated++;
            }

            if (res.next < 0) break;

            // 更新logits指针
            lg = static_cast<const uint16_t*>(
                npu_.req.get_output_tensor(0).data());
            next_token = res.next;

            // 检查停止条件
            if (!stop.isEmpty() && generated_text.contains(stop)) {
                stopped_by_sequence = true;
                break;
            }
        }

        // 处理缓冲区中剩余的字节（可能包含不完整的UTF-8序列）
        if (!utf8_buffer.isEmpty()) {
            // 尝试将剩余字节作为完整序列处理
            // 如果不是有效的UTF-8，使用replacement character
            QString remaining = QString::fromUtf8(utf8_buffer);
            if (!remaining.isEmpty()) {
                generated_text += remaining;
                if (stream_callback) {
                    stream_callback(remaining);
                }
            }
        }

        GenerateResult result;
        result.text = generated_text;
        result.prompt_tokens = prompt_tokens;
        result.completion_tokens = generated;
        result.finish_reason = !stopped_by_chat && !stopped_by_sequence &&
            generated >= max_tokens ? "length" : "stop";

        return result;
    }

private:
    // 构建prompt
    std::string buildPrompt(const std::vector<ChatMessage>& messages) {
        std::string prompt;
        for (size_t i = 0; i < messages.size(); i++) {
            const ChatMessage& msg = messages[i];
            if (msg.role == "system") {
                prompt += msg.content.toStdString() + "\n\n";
            } else if (msg.role == "user") {
                prompt += "User: " + msg.content.toStdString() + "\n\nAssistant:";
            } else if (msg.role == "assistant") {
                prompt += msg.content.toStdString() + "\n\n";
            }
        }
        return prompt;
    }

    // 核心组件
    RWKVWeights* draft_ = nullptr;
    RWKVState* state_ = nullptr;
    NpuWindow npu_;
    EmbeddingTable emb_;
    std::unique_ptr<SpecSampler> sampler_;
    std::unique_ptr<StateCache> cache_;
    rwkv::TrieTokenizer tokenizer_;
    uint64_t rng_;

    // 模型信息
    int dim_ = 0;
    int vocab_ = 0;
    int n_layers_ = 0;

    // 互斥锁
    QMutex mutex_;
};
