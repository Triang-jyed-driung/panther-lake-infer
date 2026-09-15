#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <cstring>

#include "vocab_data.hpp"

namespace rwkv {

class TrieTokenizer {
private:
    // 单字节token: byte -> token_id (硬规则: byte+1)
    // 不需要查表，直接 byte+1

    // Trie节点
    struct Node {
        // children用vector存储，支持任意数量子节点
        struct Child { uint8_t byte; uint32_t node; };
        std::vector<Child> ch;
        int32_t tok;  // -1表示不是token

        Node() : tok(-1) {}

        uint32_t find(uint8_t b) const {
            for (size_t i = 0; i < ch.size(); ++i)
                if (ch[i].byte == b) return ch[i].node;
            return 0;
        }
        void add(uint8_t b, uint32_t n) { ch.push_back({b, n}); }
    };

    std::vector<Node> nodes_;  // 所有trie节点

    // 65536密集索引: (byte0<<8)|byte1 -> trie根节点索引 (0表示空)
    uint32_t idx_[65536];

    // 在trie中查找最长匹配，返回(token_id, 消耗字节数)
    std::pair<int32_t, int> trie_find(uint32_t root, const uint8_t* data, int len) const {
        int32_t best = -1;
        int best_len = 0;
        uint32_t node = root;
        for (int i = 0; i < len; ++i) {
            node = nodes_[node].find(data[i]);
            if (!node) break;
            if (nodes_[node].tok >= 0) {
                best = nodes_[node].tok;
                best_len = i + 1;
            }
        }
        return {best, best_len};
    }

public:
    TrieTokenizer() {
        memset(idx_, 0, sizeof(idx_));
        nodes_.reserve(262144);

        // 第一遍：收集每个2字节前缀下的token
        struct PT { int32_t self = -1; std::vector<std::pair<uint8_t, int32_t>> ch; };
        std::vector<PT> pt(65536);

        for (uint32_t i = 0; i < VOCAB_COUNT; ++i) {
            const uint8_t* d = &VOCAB_DATA[VOCAB_OFFSETS[i]];
            uint8_t len = d[0];
            const uint8_t* b = d + 1;
            int32_t tid = i + 1;

            // 单字节token: byte+1，不需要存储
            if (len == 1) continue;

            uint16_t key = (b[0] << 8) | b[1];
            if (len == 2) { pt[key].self = tid; }
            else          { pt[key].ch.push_back({b[2], tid}); }
        }

        // 第二遍：为每个前缀构建trie
        for (uint32_t k = 0; k < 65536; ++k) {
            if (pt[k].ch.empty() && pt[k].self < 0) continue;

            // 创建根节点
            uint32_t root = nodes_.size();
            nodes_.emplace_back();
            nodes_[root].tok = pt[k].self;
            idx_[k] = root + 1; // +1 以便0表示空

            // 插入children
            for (auto& [byte, tid] : pt[k].ch) {
                // 这个token剩下的字节: byte是第3字节，后面可能还有
                // 但我们只存了(tid, byte)，需要从VOCAB_DATA恢复完整token
                // 实际上tid对应的token是完整的，我们需要知道第3字节之后的部分
                // 重新从VOCAB_DATA获取
                const uint8_t* td = &VOCAB_DATA[VOCAB_OFFSETS[tid - 1]];
                uint8_t tlen = td[0];
                const uint8_t* tb = td + 1;

                uint32_t node = root;
                for (uint8_t j = 2; j < tlen; ++j) { // 从第3字节开始
                    uint32_t child = nodes_[node].find(tb[j]);
                    if (!child) {
                        child = nodes_.size();
                        nodes_.emplace_back();
                        nodes_[node].add(tb[j], child);
                    }
                    node = child;
                }
                nodes_[node].tok = tid;
            }
        }
    }

    std::vector<int32_t> encode(const std::string& text) const {
        return encode_bytes(reinterpret_cast<const uint8_t*>(text.data()), text.size());
    }

    std::vector<int32_t> encode_bytes(const uint8_t* data, size_t len) const {
        std::vector<int32_t> tok;
        tok.reserve(len);
        const uint8_t* p = data;
        const uint8_t* e = data + len;

        while (p < e) {
            if (p + 1 < e) {
                uint32_t root = idx_[(p[0] << 8) | p[1]];
                if (root) {
                    // 根节点可能有token（长度为2的token）
                    int32_t best = nodes_[root - 1].tok;
                    int best_used = 0;
                    
                    auto [tid, used] = trie_find(root - 1, p + 2, e - p - 2);
                    if (tid >= 0) {
                        best = tid;
                        best_used = used;
                    }
                    
                    if (best >= 0) {
                        tok.push_back(best);
                        p += 2 + best_used;
                        continue;
                    }
                }
            }
            // fallback: 单字节token = byte+1
            tok.push_back(p[0] + 1);
            ++p;
        }
        return tok;
    }

    std::vector<uint8_t> decode_bytes(const std::vector<int32_t>& tokens) const {
        std::vector<uint8_t> out;
        for (int32_t t : tokens) {
            if (t >= 1 && t <= (int32_t)VOCAB_COUNT) {
                const uint8_t* d = &VOCAB_DATA[VOCAB_OFFSETS[t - 1]];
                uint8_t len = d[0];
                out.insert(out.end(), d + 1, d + 1 + len);
            }
        }
        return out;
    }

    std::string decode(const std::vector<int32_t>& tokens) const {
        auto b = decode_bytes(tokens);
        return std::string(b.begin(), b.end());
    }

    size_t vocab_size() const { return VOCAB_COUNT; }
    size_t memory_usage() const {
        return sizeof(*this) + sizeof(VOCAB_DATA) + sizeof(VOCAB_OFFSETS)
             + nodes_.capacity() * sizeof(Node);
    }
};

} // namespace rwkv