/*
 * state_cache.h — RWKV 状态 LRU 缓存
 * =====================================
 *
 * 基于token序列hash的状态缓存
 * - 状态存储在 /tmp/rwkv_states/
 * - 文件名：hash(所有token).state
 * - LRU淘汰，最多K个状态
 *
 * 状态布局（二进制）：
 *   [20字节]  header: magic/version + state sizes + logits_size
 *   [N字节]   draft model state (RWKVState)
 *   [M字节]   NPU VariableState (每层: xp[D] + wkv[H*hs*hs] + fp[D])
 *   [logits_size * 2字节] last valid prompt row logits (fp16)
 */
#pragma once

#include <QObject>
#include <QHash>
#include <QList>
#include <QByteArray>
#include <QDir>
#include <QFile>
#include <QSaveFile>
#include <QDateTime>
#include <QMutex>
#include <QMutexLocker>
#include <QCryptographicHash>
#include <vector>
#include <cstdint>
#include <cstring>

#include "rwkv.h"
#include "npu/npu_window.h"

// 状态条目
struct StateEntry {
    QString filepath;        // 状态文件路径
    qint64 last_access;      // 最后访问时间
    int token_count;         // token数量
};

// 状态缓存
class StateCache : public QObject {
    Q_OBJECT

public:
    explicit StateCache(int max_states = 3, QObject *parent = nullptr)
        : QObject(parent), max_states_(max_states) {
        cache_dir_ = "/tmp/rwkv_states";
        QDir().mkpath(cache_dir_);
    }

    ~StateCache() {
        // 可选：清理临时文件
    }

    // Cache identity only; selected for incremental prefix-hash performance.
    static constexpr auto HASH_ALGORITHM = QCryptographicHash::Sha1;

    // 计算token序列的hash
    static QByteArray hashTokens(const std::vector<int>& tokens) {
        QCryptographicHash hash(HASH_ALGORITHM);
        hash.addData(QByteArray((const char*)tokens.data(), tokens.size() * sizeof(int)));
        return hash.result().toHex();  // 完整SHA-1作为文件名
    }

    // 检查缓存是否存在
    bool contains(const std::vector<int>& tokens) {
        QMutexLocker lock(&mutex_);
        QByteArray hash = hashTokens(tokens);
        return cache_map_.contains(hash);
    }
    
    // 直接用hash查找（避免重复计算）
    bool findByHash(const QByteArray& hash, QString& filepath) {
        QMutexLocker lock(&mutex_);
        auto it = cache_map_.find(hash);
        if (it == cache_map_.end()) return false;
        
        // 更新访问时间
        it.value().last_access = QDateTime::currentMSecsSinceEpoch();
        filepath = it.value().filepath;
        
        // 移到LRU列表前端
        lru_list_.removeOne(hash);
        lru_list_.prepend(hash);
        
        return true;
    }

    // 查找缓存状态
    bool find(const std::vector<int>& tokens, QString& filepath) {
        QMutexLocker lock(&mutex_);
        QByteArray hash = hashTokens(tokens);

        auto it = cache_map_.find(hash);
        if (it == cache_map_.end()) return false;

        // 更新访问时间
        it.value().last_access = QDateTime::currentMSecsSinceEpoch();
        filepath = it.value().filepath;

        // 移到LRU列表前端
        lru_list_.removeOne(hash);
        lru_list_.prepend(hash);

        return true;
    }

    // 保存状态到缓存
    bool save(const std::vector<int>& tokens,
              RWKVWeights* weights,
              RWKVState* draft_state,
              NpuWindow& npu,
              const std::vector<uint16_t>& logits) {
        QMutexLocker lock(&mutex_);
        if (max_states_ <= 0 || logits.size() != (size_t)npu.V) return false;
        QByteArray hash = hashTokens(tokens);
        QString filepath = cache_dir_ + "/" + hash + ".state";

        // 淘汰旧状态
        while (!cache_map_.contains(hash) && cache_map_.size() >= max_states_) {
            evictOldest();
        }

        // 写入状态文件
        if (!writeStateFile(filepath, weights, draft_state, npu, logits)) {
            return false;
        }

        // 添加到缓存
        StateEntry entry;
        entry.filepath = filepath;
        entry.last_access = QDateTime::currentMSecsSinceEpoch();
        entry.token_count = (int)tokens.size();

        cache_map_[hash] = entry;
        lru_list_.removeOne(hash);
        lru_list_.prepend(hash);

        return true;
    }

    // 加载状态
    bool load(const std::vector<int>& tokens,
              RWKVState* draft_state,
              NpuWindow& npu,
              std::vector<uint16_t>& logits) {
        QMutexLocker lock(&mutex_);
        QByteArray hash = hashTokens(tokens);

        auto it = cache_map_.find(hash);
        if (it == cache_map_.end()) return false;

        QString filepath = it.value().filepath;
        bool ok = readStateFile(filepath, draft_state, npu, logits);

        if (ok) {
            // 更新访问时间
            it.value().last_access = QDateTime::currentMSecsSinceEpoch();
            lru_list_.removeOne(hash);
            lru_list_.prepend(hash);
        }

        return ok;
    }
    
    // 直接用hash加载状态（避免重复计算）
    bool loadByHash(const QByteArray& hash,
                    RWKVState* draft_state,
                    NpuWindow& npu,
                    std::vector<uint16_t>& logits) {
        QMutexLocker lock(&mutex_);

        auto it = cache_map_.find(hash);
        if (it == cache_map_.end()) return false;

        QString filepath = it.value().filepath;
        bool ok = readStateFile(filepath, draft_state, npu, logits);

        if (ok) {
            // 更新访问时间
            it.value().last_access = QDateTime::currentMSecsSinceEpoch();
            lru_list_.removeOne(hash);
            lru_list_.prepend(hash);
        }

        return ok;
    }

    // 清除所有缓存
    void clear() {
        QMutexLocker lock(&mutex_);
        for (const auto& entry : cache_map_) {
            QFile::remove(entry.filepath);
        }
        cache_map_.clear();
        lru_list_.clear();
    }

    int size() const {
        QMutexLocker lock(&mutex_);
        return cache_map_.size();
    }
    
    // 获取已缓存状态中最大的token数量
    int maxCachedTokens() const {
        QMutexLocker lock(&mutex_);
        int max_tokens = 0;
        for (auto it = cache_map_.begin(); it != cache_map_.end(); ++it) {
            if (it.value().token_count > max_tokens)
                max_tokens = it.value().token_count;
        }
        return max_tokens;
    }

private:
    // 淘汰最久未访问的状态
    void evictOldest() {
        if (lru_list_.isEmpty()) return;

        QByteArray hash = lru_list_.takeLast();
        auto it = cache_map_.find(hash);
        if (it != cache_map_.end()) {
            QFile::remove(it.value().filepath);
            cache_map_.erase(it);
        }
    }

    // 状态文件格式常量
    static constexpr uint32_t STATE_MAGIC = 0x52574B56;  // "RWKV"
    static constexpr uint32_t STATE_VERSION = 3;

    struct StateFileHeader {
        uint32_t magic;
        uint32_t version;
        uint32_t draft_state_size;   // floats
        uint32_t npu_state_size;     // bytes
        uint32_t logits_size;       // fp16 elements
    };

    // 写入状态文件
    // NPU状态直接从VariableState内部缓冲区读取（已累积的状态）
    bool writeStateFile(const QString& filepath,
                        RWKVWeights* weights,
                        RWKVState* draft_state,
                        NpuWindow& npu,
                        const std::vector<uint16_t>& logits) {
        QSaveFile file(filepath);
        if (!file.open(QIODevice::WriteOnly)) return false;

        // 准备header
        StateFileHeader header;
        header.magic = STATE_MAGIC;
        header.version = STATE_VERSION;
        header.draft_state_size = (uint32_t)rwkv_state_size(weights);
        header.npu_state_size = (uint32_t)((size_t)npu.L *
            (2 * (size_t)npu.D + (size_t)npu.H * npu.hs * npu.hs) * sizeof(float));
        header.logits_size = (uint32_t)logits.size();

        // 写header
        if (file.write((const char*)&header, sizeof(header)) != sizeof(header)) return false;

        // 写draft state
        size_t draft_bytes = header.draft_state_size * sizeof(float);
        std::vector<float> draft_buf(header.draft_state_size);
        if (draft_state) {
            rwkv_state_save(draft_state, draft_buf.data());
        }
        if (file.write((const char*)draft_buf.data(), draft_bytes) != (qint64)draft_bytes) return false;

        // 写NPU状态 - 直接从VariableState内部缓冲区读取
        // 每层3个状态：xp[D], wkv[H*hs*hs], fp[D]
        for (int l = 0; l < npu.L; l++) {
            // att_x_prev [D]
            int idx_xp = npu.name2idx[l][0];
            const qint64 row_bytes = (qint64)npu.D * sizeof(float);
            if (file.write((const char*)npu.st_ptr[idx_xp], row_bytes) != row_bytes) return false;
            
            // att_state [H, hs, hs]
            int idx_wkv = npu.name2idx[l][1];
            const qint64 wkv_bytes = (qint64)npu.H * npu.hs * npu.hs * sizeof(float);
            if (file.write((const char*)npu.st_ptr[idx_wkv], wkv_bytes) != wkv_bytes) return false;
            
            // ffn_x_prev [D]
            int idx_fp = npu.name2idx[l][2];
            if (file.write((const char*)npu.st_ptr[idx_fp], row_bytes) != row_bytes) return false;
        }

        const qint64 logits_bytes = (qint64)logits.size() * sizeof(uint16_t);
        if (file.write((const char*)logits.data(), logits_bytes) != logits_bytes) return false;
        return file.commit();
    }

    // 读取状态文件
    // 直接恢复到VariableState内部缓冲区
    bool readStateFile(const QString& filepath,
                       RWKVState* draft_state,
                       NpuWindow& npu,
                       std::vector<uint16_t>& logits) {
        QFile file(filepath);
        if (!file.open(QIODevice::ReadOnly)) return false;

        // 读header
        StateFileHeader header;
        if (file.read((char*)&header, sizeof(header)) != sizeof(header))
            return false;

        if (header.magic != STATE_MAGIC || header.version != STATE_VERSION)
            return false;
        if (header.logits_size != (uint32_t)npu.V)
            return false;
        const size_t npu_floats = (size_t)npu.L *
            (2 * (size_t)npu.D + (size_t)npu.H * npu.hs * npu.hs);
        if (header.npu_state_size != npu_floats * sizeof(float)) return false;

        // 读draft state
        size_t draft_bytes = header.draft_state_size * sizeof(float);
        std::vector<float> draft_buf(header.draft_state_size);
        if (file.read((char*)draft_buf.data(), draft_bytes) != (qint64)draft_bytes)
            return false;

        // Read the whole entry before changing live state or logits.
        std::vector<float> npu_buf(npu_floats);
        const qint64 npu_bytes = (qint64)npu_floats * sizeof(float);
        if (file.read((char*)npu_buf.data(), npu_bytes) != npu_bytes) return false;
        std::vector<uint16_t> restored_logits(header.logits_size);
        const qint64 logits_bytes = (qint64)restored_logits.size() * sizeof(uint16_t);
        if (file.read((char*)restored_logits.data(), logits_bytes) != logits_bytes) return false;

        if (draft_state) rwkv_state_restore(draft_state, draft_buf.data());
        const float* src = npu_buf.data();

        // 读NPU状态 - 直接恢复到VariableState内部缓冲区
        for (int l = 0; l < npu.L; l++) {
            // att_x_prev [D]
            int idx_xp = npu.name2idx[l][0];
            memcpy(npu.st_ptr[idx_xp], src, npu.D * sizeof(float));
            src += npu.D;
            
            // att_state [H, hs, hs]
            int idx_wkv = npu.name2idx[l][1];
            const size_t wkv_size = (size_t)npu.H * npu.hs * npu.hs;
            memcpy(npu.st_ptr[idx_wkv], src, wkv_size * sizeof(float));
            src += wkv_size;
            
            // ffn_x_prev [D]
            int idx_fp = npu.name2idx[l][2];
            memcpy(npu.st_ptr[idx_fp], src, npu.D * sizeof(float));
            src += npu.D;
        }

        file.close();
        logits.swap(restored_logits);
        return true;
    }

    mutable QMutex mutex_;
    int max_states_;
    QString cache_dir_;
    QHash<QByteArray, StateEntry> cache_map_;
    QList<QByteArray> lru_list_;  // LRU列表，前端最新
};
