// npu_window.h — v5d 窗口图 NPU 侧封装（单副本加载 + 状态管理），spec/state_check 共用
#pragma once
// =================================================================================
// 状态写入约定：不用 VariableState::set_state（外部张量会破坏插件参数表，
// 触发 infer 时 range_check 98>=97），而是启动时缓存 get_state().data() 指针
// （与插件 reset() 的 memset 同路径），回退 = 原地 memcpy。
// =================================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <array>
#include <algorithm>
#include <fstream>
#include <stdexcept>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/variable_state.hpp"

#include "common.h"

struct NpuWindow {
    bool gpu = false;
    void* buf = nullptr;
    size_t blob_size = 0;
    ov::Core core;
    ov::Tensor blob_tensor;
    ov::CompiledModel compiled;
    ov::InferRequest req;
    std::vector<ov::VariableState> states;       // 3L 个
    std::vector<std::array<int, 3>> name2idx;    // 每层 (xp, wkv, fp) 的 states 下标
    std::vector<float*> st_ptr;                  // 各状态内部缓冲指针（启动时缓存，稳定）
    int n = 0, D = 0, V = 0, L = 0, H = 0, hs = 0;

    // 加载 blob 文件（单副本 mmap + import_model）
    void load(const char* blob_path) {
        const char* device = std::getenv("RWKV_DEVICE");
        gpu = device && std::string(device) == "GPU";
        if (device && std::string(device) != "GPU" && std::string(device) != "NPU")
            throw std::invalid_argument("RWKV_DEVICE must be NPU or GPU");
        if (gpu) {
            std::ifstream stream(blob_path, std::ios::binary);
            if (!stream) throw std::runtime_error("Cannot open GPU blob");
            auto t0 = now_ms();
            ov::AnyMap properties;
            const char* precision = std::getenv("RWKV_GPU_PRECISION");
            if (precision) {
                if (std::string(precision) != "f32" && std::string(precision) != "f16")
                    throw std::invalid_argument("RWKV_GPU_PRECISION must be f32 or f16");
                properties["INFERENCE_PRECISION_HINT"] = std::string(precision) == "f32"
                    ? ov::element::f32 : ov::element::f16;
            }
            compiled = core.import_model(stream, "GPU", properties);
            printf("GPU import_model: %.2fs\n", (now_ms() - t0) / 1e3);
            printf("GPU inference precision: %s\n",
                   compiled.get_property("INFERENCE_PRECISION_HINT").as<ov::element::Type>().get_type_name().c_str());
        } else {
            struct stat st;
            stat(blob_path, &st);
            blob_size = (size_t)st.st_size;
            printf("blob=%s  %.2f GiB\n", blob_path, blob_size / 1073741824.0);

            buf = mmap(nullptr, blob_size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (buf == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(1); }
            int fd = open(blob_path, O_RDONLY);
            for (size_t off = 0; off < blob_size; ) {
                size_t nch = std::min<size_t>(1ull << 28, blob_size - off);
                size_t r = 0;
                while (r < nch) {
                    ssize_t k = read(fd, (char*)buf + off + r, nch - r);
                    if (k <= 0) { fprintf(stderr, "blob read failed\n"); exit(1); }
                    r += (size_t)k;
                }
                posix_fadvise(fd, (off_t)off, (off_t)nch, POSIX_FADV_DONTNEED);
                off += nch;
            }
            close(fd);

            blob_tensor = ov::Tensor(ov::element::u8, ov::Shape{blob_size}, buf);
            auto t0 = now_ms();
            compiled = core.import_model(blob_tensor, "NPU");
            printf("import_model: %.2fs\n", (now_ms() - t0) / 1e3);
            double alloc = core.get_property("NPU", "NPU_DEVICE_ALLOC_MEM_SIZE").as<uint64_t>() / 1073741824.0;
            printf("NPU_DEVICE_ALLOC_MEM_SIZE = %.2f GiB\n", alloc);

        }

        auto in0 = compiled.input(0);
        n = (int)in0.get_shape()[0];
        D = (int)in0.get_shape()[1];
        V = (int)compiled.output(0).get_shape()[1];
        L = (int)compiled.output(1).get_shape()[0];
        H = (int)compiled.output(2).get_shape()[2];
        hs = (int)compiled.output(2).get_shape()[3];

        req = compiled.create_infer_request();
        states = req.query_state();
        name2idx.assign(L, {-1, -1, -1});
        st_ptr.resize(states.size());
        for (size_t i = 0; i < states.size(); i++) {
            const std::string& nm = states[i].get_name();
            int l = atoi(nm.c_str() + 1);                  // "L12.xp" → 12
            int kind = (nm.find(".xp") != std::string::npos) ? 0
                     : (nm.find(".wkv") != std::string::npos) ? 1 : 2;
            name2idx[l][kind] = (int)i;
            if (!gpu) st_ptr[i] = (float*)states[i].get_state().data();   // 内部缓冲，稳定
        }
        printf("window: n=%d D=%d V=%d L=%d H=%d hs=%d  states=%zu\n",
               n, D, V, L, H, hs, states.size());
    }

    // 全部状态清零
    void reset_state() {
        for (size_t i = 0; i < states.size(); i++) {
            if (gpu) states[i].reset();
            else memset(st_ptr[i], 0, states[i].get_state().get_byte_size());
        }
    }

    // 快照第 col 列 → 状态缓冲（原地 memcpy，回退用）
    // att [L,n,D]，wkv [L,n,H,hs,hs]，ffn [L,n,D]，均 f32
    void rollback_snapshot(const float* att, const float* wkv,
                           const float* ffn, int col) {
        if (gpu) {
            // GPU state tensors may be copies; explicitly upload each rollback.
            for (int l = 0; l < L; l++) {
                const float* sources[] = {
                    att + ((size_t)l * n + col) * D,
                    wkv + ((size_t)l * n + col) * H * hs * hs,
                    ffn + ((size_t)l * n + col) * D};
                for (int kind = 0; kind < 3; kind++) {
                    auto& state = states[name2idx[l][kind]];
                    auto tensor = state.get_state();
                    memcpy(tensor.data(), sources[kind], tensor.get_byte_size());
                    state.set_state(tensor);
                }
            }
            return;
        }
        for (int l = 0; l < L; l++) {
            memcpy(st_ptr[name2idx[l][0]], att + ((size_t)l * n + col) * D,
                   (size_t)D * 4);
            memcpy(st_ptr[name2idx[l][1]], wkv + ((size_t)l * n + col) * H * hs * hs,
                   (size_t)H * hs * hs * 4);
            memcpy(st_ptr[name2idx[l][2]], ffn + ((size_t)l * n + col) * D,
                   (size_t)D * 4);
        }
    }

    // 便捷方法：用当前输出张量回退到第 col 列
    void rollback_to(int col) {
        rollback_snapshot(
            static_cast<const float*>(req.get_output_tensor(1).data()),
            static_cast<const float*>(req.get_output_tensor(2).data()),
            static_cast<const float*>(req.get_output_tensor(3).data()),
            col);
    }

    // 便捷方法：填入嵌入行到输入张量的指定 slot
    void fill_emb_slot(int slot, const EmbeddingTable& emb, int64_t tok) {
        float* row = static_cast<float*>(req.get_input_tensor(0).data())
                     + (size_t)slot * D;
        emb.fill_row(row, tok);
    }
};
