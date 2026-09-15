/*
 * chat_callback.h — 聊天回调自动机
 * ===================================
 *
 * 聊天模板：
 *   User: something?\n\n
 *   Assistant: Answer\n\n
 *   User: another round\n\n
 *   Assistant: ...
 *
 * callback函数自动机状态：
 *   状态0：正常输出
 *     - 如果token有两个连续换行(\n\n)，直接进入状态2停止
 *     - 如果末尾有一个换行，进入状态1
 *     - 否则保持状态0
 *
 *   状态1：前一个token末尾有换行
 *     - 如果token开头是换行，进入状态2停止
 *     - 否则和状态0一样处理
 *
 *   状态2：停止状态
 *
 * 使用方法：
 *   ChatCallback cb(tokenizer);
 *   // 在投机解码的callback中调用cb.on_token(token_id)，返回true表示应该停止
 *   bool should_stop = cb.on_token(token_id);
 *   // 重置状态（开始新的回复）
 *   cb.reset();
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include "tokenizer/rwkv_tokenizer.hpp"

class ChatCallback {
public:
    // 状态定义
    enum class State {
        NORMAL = 0,      // 正常输出
        LAST_NEWLINE = 1, // 上一个token末尾有换行
        STOP = 2         // 停止
    };

private:
    State state_;
    const rwkv::TrieTokenizer* tokenizer_;
    std::string current_text_;  // 当前累积的文本

public:
    explicit ChatCallback(const rwkv::TrieTokenizer* tokenizer)
        : state_(State::NORMAL), tokenizer_(tokenizer) {}

    // 重置状态（开始新的回复）
    void reset() {
        state_ = State::NORMAL;
        current_text_.clear();
    }

    // 获取当前状态
    State state() const { return state_; }

    // 获取累积的文本
    const std::string& current_text() const { return current_text_; }

    // 处理token，返回true表示应该停止
    bool on_token(int32_t token_id) {
        // 如果已经是停止状态，直接返回
        if (state_ == State::STOP) {
            return true;
        }

        // token 0 是特殊的停止token
        if (token_id == 0) {
            state_ = State::STOP;
            return true;
        }

        // 解码token为文本
        std::vector<int32_t> single_tok = {token_id};
        std::string token_text = tokenizer_->decode(single_tok);
        current_text_ += token_text;

        // 根据当前状态处理
        switch (state_) {
            case State::NORMAL: {
                // 检查是否有两个连续换行
                if (token_text.find("\n\n") != std::string::npos) {
                    state_ = State::STOP;
                    return true;
                }
                // 检查末尾是否有换行
                if (!token_text.empty() && token_text.back() == '\n') {
                    state_ = State::LAST_NEWLINE;
                    return false;
                }
                // 保持状态0
                return false;
            }

            case State::LAST_NEWLINE: {
                // 检查开头是否有换行
                if (!token_text.empty() && token_text[0] == '\n') {
                    state_ = State::STOP;
                    return true;
                }
                // 检查是否有换行（前一个token末尾有换行 + 当前token有换行 = 两个连续换行）
                if (token_text.find('\n') != std::string::npos) {
                    state_ = State::STOP;
                    return true;
                }
                // 和状态0一样处理
                if (!token_text.empty() && token_text.back() == '\n') {
                    state_ = State::LAST_NEWLINE;
                } else {
                    state_ = State::NORMAL;
                }
                return false;
            }

            case State::STOP:
                return true;

            default:
                return false;
        }
    }
};
