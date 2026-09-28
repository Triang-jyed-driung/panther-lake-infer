// common.h — 跨文件共享工具（npu/ 目录下 spec、state_check、npu_run 共用）
#pragma once

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <omp.h>

#ifdef __linux__
#include <sched.h>
#endif

#include "../gemm_fp16.h"
#include "../rwkv_tune.h"

// fp16 位型 → float（可移植，不依赖硬件 F16C）
static inline float f16_to_f32(uint16_t h) {
    uint32_t b = (uint32_t)(h & 0x8000) << 16
               | ((h & 0x7C00) + 0x1C000) << 13
               | (h & 0x03FF) << 13;
    float v; memcpy(&v, &b, 4); return v;
}

// 高精度计时（毫秒）
static inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// CPU环境初始化
static void setup_cpu(int n_cores = RWKV_BIND_CORES) {
    rwkv_tune_once(n_cores);
}

// 融合嵌入表封装（mmap 只读，[V,D] fp16 连续布局）
struct EmbeddingTable {
    uint16_t* data = nullptr;
    int V = 0, D = 0;
    size_t map_size = 0;

    // 从文件加载嵌入表；expected_D 用于校验文件大小 = V * D * 2
    static EmbeddingTable load(const char* path, int expected_D) {
        struct stat st;
        if (stat(path, &st) != 0) {
            fprintf(stderr, "cannot stat %s\n", path);
            return {};
        }
        size_t sz = (size_t)st.st_size;
        int D = expected_D;
        if (D <= 0 || sz % (2 * (size_t)D) != 0) {
            fprintf(stderr, "emb table size mismatch\n");
            return {};
        }
        int V = (int)(sz / 2 / D);
        int fd = open(path, O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "open %s failed\n", path);
            return {};
        }
        uint16_t* p = (uint16_t*)mmap(nullptr, sz, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (p == MAP_FAILED) {
            fprintf(stderr, "emb mmap failed\n");
            return {};
        }
        return {p, V, D, sz};
    }

    // 查行：token → [D] float（fp16 值的 f32 表示）
    void fill_row(float* dst, int64_t tok) const {
        const uint16_t* src = data + (size_t)tok * D;
        for (int j = 0; j < D; j++) dst[j] = f16_to_f32(src[j]);
    }

    // 释放 mmap
    void unload() {
        if (data && data != MAP_FAILED) {
            munmap(data, map_size);
            data = nullptr;
        }
    }

    explicit operator bool() const { return data != nullptr; }
};
