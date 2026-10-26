// zero_window.h — NpuWindow 的裸 Level Zero 实现（drop-in 替代 npu_window.h）
// =========================================================================
// 目标：跑一个【已编译】的 NPU blob（窗口图），完全不依赖 libopenvino.so。
// 唯一外部依赖：libze_loader.so（Level Zero loader，系统库）+ intel_vpu 内核驱动。
//
// 为什么不直接链 OpenVINO：跑一个自包含的预编译 blob，OpenVINO 的 C++ API
// (ov::Output<Node>::get_shape) 会把整套 core IR（数百个文件）从符号闭包拖进来。
// 而 blob 的 shape/名字都在驱动元数据里，IR 对本用例纯属死重。这里直接调
// ZE_extension_graph 的图扩展 API，接口与原 npu_window.h 一致，spec.cpp/chat.cpp 几乎不改。
//
// 图契约（见 scripts/npu/build_window_graph.py）：
//   in [0]        emb_x      [n, D] f32        —— 融合 ln0 后的词嵌入行
//   out[0]        logits     [n, V] f16
//   out[1]        att_snaps  [L, n, D] f32
//   out[2]        wkv_snaps  [L, n, H, hs, hs] f32
//   out[3]        ffn_snaps  [L, n, D] f32
//   状态（ReadValue/Assign 成对，绑同一缓冲）：每层 {L}.xp / {L}.wkv / {L}.fp，共 3L
//
// 状态按【名字】映射（保留原 rollback 语义）：
//   驱动里状态 arg 名 = "vpux_ie_read_value_L12.xp"（输入）/ "vpux_ie_assign_L12.xp"（输出）
//   剥前缀得 "L12.xp" → l=12, kind=xp(0)/wkv(1)/fp(2) → name2idx[l][kind]
//   若某状态元数据缺名（老 blob），fallback 到 l*3+kind 索引序。
// =========================================================================
#pragma once

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>

#include <level_zero/ze_api.h>
#include <ze_graph_ext.h>

#include "common.h"  // EmbeddingTable, now_ms

// ---------------------------------------------------------------------------
// 迷你 ov:: shim —— 仅为让 spec.cpp 的 set_input_tensor / get_*_tensor 调用原样编译。
// 只承载一个宿主指针，不做任何 OpenVINO 语义。
// ---------------------------------------------------------------------------
namespace ov {
namespace element {
struct Type {};
inline constexpr Type f32{}, f16{}, u8{}, i64_{};
}  // namespace element

struct Shape : std::vector<size_t> {
    using std::vector<size_t>::vector;
};

struct Tensor {
    void* data_ = nullptr;
    size_t bytes_ = 0;
    Tensor() = default;
    Tensor(element::Type, const Shape& s, void* d) : data_(d) {
        bytes_ = 1;
        for (size_t v : s) bytes_ *= v;
    }
    void* data() const { return data_; }
    size_t get_byte_size() const { return bytes_; }
};
}  // namespace ov

namespace zwin {

// ── 小工具 ────────────────────────────────────────────────────────────────
#define ZCK(call)                                                            \
    do {                                                                     \
        ze_result_t _r = (call);                                             \
        if (_r != ZE_RESULT_SUCCESS) {                                       \
            fprintf(stderr, "[zero] %s:%d  %s -> ze_result=%d\n", __FILE__,  \
                    __LINE__, #call, (int)_r);                               \
            throw std::runtime_error("Level Zero call failed");              \
        }                                                                    \
    } while (0)

inline bool prefix_of(const std::string& s, const char* p) {
    return s.rfind(p, 0) == 0;
}
inline size_t elem_size(ze_graph_argument_precision_t pr) {
    switch (pr) {
        case ZE_GRAPH_ARGUMENT_PRECISION_FP64:
        case ZE_GRAPH_ARGUMENT_PRECISION_UINT64:
        case ZE_GRAPH_ARGUMENT_PRECISION_INT64: return 8;
        case ZE_GRAPH_ARGUMENT_PRECISION_FP32:
        case ZE_GRAPH_ARGUMENT_PRECISION_UINT32:
        case ZE_GRAPH_ARGUMENT_PRECISION_INT32: return 4;
        case ZE_GRAPH_ARGUMENT_PRECISION_FP16:
        case ZE_GRAPH_ARGUMENT_PRECISION_BF16:
        case ZE_GRAPH_ARGUMENT_PRECISION_UINT16:
        case ZE_GRAPH_ARGUMENT_PRECISION_INT16: return 2;
        default: return 1;
    }
}

// 图参数的可移植描述
struct ArgInfo {
    uint32_t index = 0;
    ze_graph_argument_type_t type = ZE_GRAPH_ARGUMENT_TYPE_INPUT;
    std::string name;                 // 驱动名（可能带 vpux_ie_* 前缀）
    std::string logical;              // 剥前缀后的名（状态用）
    bool is_state = false;
    std::vector<size_t> shape;        // 来自 GetArgumentMetadata.shape_size
    size_t bytes = 0;
    void* buf = nullptr;              // zeMemAllocShared 分配的宿主可访问缓冲
};

// ── L0 后端句柄 ───────────────────────────────────────────────────────────
struct L0Context {
    ze_driver_handle_t driver = nullptr;
    ze_device_handle_t device = nullptr;
    ze_context_handle_t ctx = nullptr;
    ze_command_queue_handle_t cq = nullptr;
    ze_command_list_handle_t cl = nullptr;
    ze_graph_dditable_ext_t* g = nullptr;   // ZE_extension_graph DDI 表
    ze_graph_handle_t graph = nullptr;

    void init();                  // 枚举 NPU 驱动/设备，建 context/queue/list，取图扩展表
    void create_graph(const void* blob, size_t size);
    ~L0Context();
};

// ── NpuWindow（与原 npu_window.h 同名同接口的 drop-in）────────────────────
struct NpuWindow {
    void* buf = nullptr;          // blob 映射缓冲（只读，喂给驱动）
    size_t blob_size = 0;
    L0Context l0;

    std::vector<ArgInfo> args;    // 全部图参数（含状态）
    ArgInfo* in_arg = nullptr;    // emb_x
    ArgInfo* out_logits = nullptr;
    ArgInfo* out_att = nullptr;
    ArgInfo* out_wkv = nullptr;
    ArgInfo* out_ffn = nullptr;

    std::vector<void*> st_ptr;                 // 各状态缓冲（与 states 索引一致）
    std::vector<size_t> st_bytes;
    std::vector<std::array<int, 3>> name2idx;  // 每层 (xp, wkv, fp) 的 st_ptr 下标
    int n = 0, D = 0, V = 0, L = 0, H = 0, hs = 0;

    void* pending_input = nullptr;  // set_input_tensor 传入的外部宿主缓冲

    // ---- 请求 shim：暴露与 ov::InferRequest 相同的最小面 ----
    struct Request {
        NpuWindow* self = nullptr;
        void set_input_tensor(int, const ov::Tensor& t) { self->pending_input = t.data(); }
        ov::Tensor get_input_tensor(int) {
            ov::Tensor t;
            t.data_ = self->in_arg ? self->in_arg->buf : nullptr;
            t.bytes_ = self->in_arg ? self->in_arg->bytes : 0;
            return t;
        }
        ov::Tensor get_output_tensor(int i) {
            ArgInfo* a = nullptr;
            switch (i) {
                case 0: a = self->out_logits; break;
                case 1: a = self->out_att; break;
                case 2: a = self->out_wkv; break;
                case 3: a = self->out_ffn; break;
                default: break;
            }
            ov::Tensor t;
            t.data_ = a ? a->buf : nullptr;
            t.bytes_ = a ? a->bytes : 0;
            return t;
        }
        void infer();  // 定义在后（用到 L0Context）
    } req{this};

    // ---- 生命周期 ----
    void load(const char* blob_path);   // mmap blob + 建图 + 枚举/分配/绑定参数
    void reset_state();                 // 全部状态清零
    void rollback_snapshot(const float* att, const float* wkv, const float* ffn, int col);
    void rollback_to(int col);          // 从 out[1..3] 第 col 列回退状态
    void fill_emb_slot(int slot, const EmbeddingTable& emb, int64_t tok);

private:
    void classify_and_map();            // 枚举参数 → 分类 → name2idx → 分配绑定
    void dump_args() const;
};

}  // namespace zwin

// 保持外部代码 `NpuWindow` 无命名空间前缀
using NpuWindow = zwin::NpuWindow;
