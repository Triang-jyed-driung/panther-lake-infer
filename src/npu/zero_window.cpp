// zero_window.cpp — zero_window.h 的 Level Zero 实现
#include "zero_window.h"

#include <algorithm>
#include <cmath>

namespace zwin {

// 状态前缀（与 OpenVINO prefix.hpp 一致：vpux_ie_read_value_ / vpux_ie_assign_）
static const char* READVALUE_PREFIX = "vpux_ie_read_value_";
static const char* ASSIGN_PREFIX = "vpux_ie_assign_";

// ===========================================================================
// L0Context::init —— 枚举 NPU 驱动/设备，建 context/queue/list，取图扩展 DDI 表
// ===========================================================================
void L0Context::init() {
    ZCK(zeInit(0));

    uint32_t drv_count = 0;
    ZCK(zeDriverGet(&drv_count, nullptr));
    std::vector<ze_driver_handle_t> drivers(drv_count);
    ZCK(zeDriverGet(&drv_count, drivers.data()));

    // 选支持 ZE_extension_graph 的驱动（即 NPU 驱动）
    for (auto d : drivers) {
        void* fn = nullptr;
        ze_result_t r = zeDriverGetExtensionFunctionAddress(d, ZE_GRAPH_EXT_NAME, &fn);
        if (r == ZE_RESULT_SUCCESS && fn != nullptr) {
            driver = d;
            g = reinterpret_cast<ze_graph_dditable_ext_t*>(fn);
            break;
        }
    }
    if (!driver) {
        fprintf(stderr, "[zero] 未找到支持 %s 的 NPU 驱动\n", ZE_GRAPH_EXT_NAME);
        throw std::runtime_error("no NPU driver");
    }

    uint32_t dev_count = 0;
    ZCK(zeDeviceGet(driver, &dev_count, nullptr));
    std::vector<ze_device_handle_t> devices(dev_count);
    ZCK(zeDeviceGet(driver, &dev_count, devices.data()));
    device = devices[0];  // NPU 通常只有一个设备

    ze_context_desc_t cdesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    ZCK(zeContextCreate(driver, &cdesc, &ctx));

    ze_command_queue_desc_t qdesc = {};
    qdesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    qdesc.index = 0;
    qdesc.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    qdesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ZCK(zeCommandQueueCreate(ctx, device, &qdesc, &cq));

    ze_command_list_desc_t ldesc = {};
    ldesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    ldesc.commandQueueGroupOrdinal = 0;
    ZCK(zeCommandListCreate(ctx, device, &ldesc, &cl));
}

void L0Context::create_graph(const void* blob, size_t size) {
    ze_graph_desc_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_GRAPH_DESC;
    desc.format = ZE_GRAPH_FORMAT_NATIVE;   // 预编译 blob
    desc.inputSize = size;
    desc.pInput = reinterpret_cast<const uint8_t*>(blob);
    desc.pBuildFlags = nullptr;
    ZCK(g->pfnCreate(ctx, device, &desc, &graph));
}

L0Context::~L0Context() {
    if (graph && g) g->pfnDestroy(graph);
    if (cl) zeCommandListDestroy(cl);
    if (cq) zeCommandQueueDestroy(cq);
    if (ctx) zeContextDestroy(ctx);
}

// ===========================================================================
// classify_and_map —— 枚举图参数 → 分类 → name2idx → zeMemAllocShared 绑定
// ===========================================================================
void NpuWindow::classify_and_map() {
    ze_graph_properties_t props = {};
    props.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    ZCK(l0.g->pfnGetProperties(l0.graph, &props));
    const uint32_t nargs = props.numGraphArgs;
    args.resize(nargs);

    in_arg = out_logits = out_att = out_wkv = out_ffn = nullptr;
    std::vector<ArgInfo*> state_ins, state_outs;

    for (uint32_t i = 0; i < nargs; i++) {
        ArgInfo& a = args[i];
        a.index = i;

        ze_graph_argument_properties_t ap = {};
        ap.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
        ZCK(l0.g->pfnGetArgumentProperties(l0.graph, i, &ap));
        a.type = ap.type;
        a.name = ap.name;

        // 形状/字节数：用 GetArgumentMetadata 的 shape_size（v1 无显式 rank）
        ze_graph_argument_metadata_t md = {};
        md.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_METADATA;
        if (l0.g->pfnGraphGetArgumentMetadata &&
            l0.g->pfnGraphGetArgumentMetadata(l0.graph, i, &md) == ZE_RESULT_SUCCESS) {
            for (uint32_t d = 0; d < md.shape_size && d < ZE_MAX_GRAPH_TENSOR_REF_DIMS; d++)
                a.shape.push_back((size_t)md.shape[d]);
        } else {
            // 回退：v1 dims（无 rank，尽量读到非零末尾）
            for (uint32_t d = 0; d < ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE; d++)
                if (ap.dims[d]) a.shape.push_back(ap.dims[d]);
        }
        size_t es = elem_size(ap.devicePrecision);
        a.bytes = es;
        for (size_t s : a.shape) a.bytes *= s;

        // 分类：状态输入/输出按前缀；普通输入=emb_x；普通输出按形状定位
        bool is_in = (a.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT);
        if (is_in && prefix_of(a.name, READVALUE_PREFIX)) {
            a.is_state = true;
            a.logical = a.name.substr(strlen(READVALUE_PREFIX));
            state_ins.push_back(&a);
        } else if (!is_in && prefix_of(a.name, ASSIGN_PREFIX)) {
            a.is_state = true;
            a.logical = a.name.substr(strlen(ASSIGN_PREFIX));
            state_outs.push_back(&a);
        } else if (is_in) {
            in_arg = &a;   // emb_x [n,D] f32
        } else {
            // 普通输出：logits[n,V] / att[L,n,D] / wkv[L,n,H,hs,hs] / ffn[L,n,D]
            size_t rk = a.shape.size();
            if (rk == 2)              out_logits = &a;
            else if (rk == 5)         out_wkv = &a;
            else if (rk == 3 && !out_att) out_att = &a;
            else if (rk == 3)         out_ffn = &a;
        }
    }

    // 从形状提取 n,D,V,L,H,hs
    if (!in_arg || in_arg->shape.size() < 2)
        throw std::runtime_error("emb_x 输入缺失或形状异常");
    n = (int)in_arg->shape[0];
    D = (int)in_arg->shape[1];
    if (!out_logits || out_logits->shape.size() < 2)
        throw std::runtime_error("logits 输出缺失");
    V = (int)out_logits->shape[1];
    if (!out_att || !out_wkv || !out_ffn)
        throw std::runtime_error("att/wkv/ffn 快照输出缺失");
    L = (int)out_att->shape[0];
    if (out_wkv->shape.size() == 5) {
        H  = (int)out_wkv->shape[2];
        hs = (int)out_wkv->shape[3];
    }

    // 状态按名字建 name2idx（与原 npu_window.h 相同语义）
    const int nstates = (int)state_ins.size();
    name2idx.assign(L, {-1, -1, -1});
    st_ptr.assign(nstates, nullptr);
    st_bytes.assign(nstates, 0);

    for (int i = 0; i < nstates; i++) {
        ArgInfo* a = state_ins[i];
        int l = -1, kind = -1;
        if (!a->logical.empty()) {
            const char* nm = a->logical.c_str();
            l = atoi(nm + 1);                                 // "L12.xp" → 12
            if (a->logical.find(".xp") != std::string::npos)      kind = 0;
            else if (a->logical.find(".wkv") != std::string::npos) kind = 1;
            else if (a->logical.find(".fp") != std::string::npos)  kind = 2;
        }
        if (l < 0 || l >= L || kind < 0) {
            // fallback：索引序 l*3+kind
            l = i / 3;
            kind = i % 3;
        }
        if (l < L) name2idx[l][kind] = i;
    }

    // 每个状态分配一块 zeMemAllocShared 缓冲，ReadValue(读) 与 Assign(写) 绑同一块
    for (int i = 0; i < nstates; i++) {
        size_t bytes = std::max<size_t>(state_ins[i]->bytes, 1);
        // 若存在配对的 Assign 输出，其字节数应一致
        void* p = nullptr;
        ze_device_mem_alloc_desc_t ddesc = {};
        ddesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
        ze_host_mem_alloc_desc_t hdesc = {};
        hdesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
        ZCK(zeMemAllocShared(l0.ctx, &ddesc, &hdesc, bytes, 64, l0.device, &p));
        memset(p, 0, bytes);
        st_ptr[i] = p;
        st_bytes[i] = bytes;
        state_ins[i]->buf = p;
        ZCK(l0.g->pfnSetArgumentValue(l0.graph, state_ins[i]->index, p));
    }
    // Assign 输出绑到对应状态的同一块缓冲（按 logical 名配对）
    for (ArgInfo* ao : state_outs) {
        for (int i = 0; i < nstates; i++) {
            if (state_ins[i]->logical == ao->logical) {
                ao->buf = st_ptr[i];
                ZCK(l0.g->pfnSetArgumentValue(l0.graph, ao->index, st_ptr[i]));
                break;
            }
        }
    }

    // 输入/输出缓冲
    auto alloc = [&](ArgInfo* a) {
        void* p = nullptr;
        ze_device_mem_alloc_desc_t ddesc = {};
        ddesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
        ze_host_mem_alloc_desc_t hdesc = {};
        hdesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
        ZCK(zeMemAllocShared(l0.ctx, &ddesc, &hdesc, std::max<size_t>(a->bytes, 1), 64, l0.device, &p));
        memset(p, 0, a->bytes);
        a->buf = p;
        ZCK(l0.g->pfnSetArgumentValue(l0.graph, a->index, p));
    };
    if (in_arg) alloc(in_arg);
    for (ArgInfo* o : {out_logits, out_att, out_wkv, out_ffn}) if (o) alloc(o);
}

void NpuWindow::dump_args() const {
    printf("[zero] 图参数 %zu 个:  n=%d D=%d V=%d L=%d H=%d hs=%d  states=%zu\n",
           args.size(), n, D, V, L, H, hs, st_ptr.size());
    for (const ArgInfo& a : args) {
        printf("  [%2u] %-8s %-40s shape=", a.index,
               a.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT ? "in" : "out", a.name.c_str());
        for (size_t s : a.shape) printf("%zu,", s);
        printf(" %zuB%s\n", a.bytes, a.is_state ? "  [state]" : "");
    }
}

// ===========================================================================
// load —— mmap blob + 建图 + 枚举/分配/绑定参数
// ===========================================================================
void NpuWindow::load(const char* blob_path) {
    struct stat st;
    if (stat(blob_path, &st) != 0) { fprintf(stderr, "cannot stat %s\n", blob_path); exit(1); }
    blob_size = (size_t)st.st_size;
    printf("blob=%s  %.2f GiB\n", blob_path, blob_size / 1073741824.0);

    // 与原实现一致：MAP_PRIVATE|MAP_ANONYMOUS 页对齐缓冲，整块读入 + fadvise 压 page cache
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

    auto t0 = now_ms();
    l0.init();
    l0.create_graph(buf, blob_size);
    printf("create_graph: %.2fs\n", (now_ms() - t0) / 1e3);

    classify_and_map();
    dump_args();
}

// ===========================================================================
// Request::infer —— 把外部 emb 拷入输入缓冲，执行图，同步
// ===========================================================================
void NpuWindow::Request::infer() {
    NpuWindow* w = self;
    if (w->pending_input && w->in_arg && w->in_arg->buf &&
        w->pending_input != w->in_arg->buf) {
        memcpy(w->in_arg->buf, w->pending_input, w->in_arg->bytes);
    }
    ZCK(w->l0.g->pfnAppendGraphInitialize(w->l0.cl, w->l0.graph, nullptr, 0, nullptr));
    ZCK(w->l0.g->pfnAppendGraphExecute(w->l0.cl, w->l0.graph, nullptr, nullptr, 0, nullptr));
    ZCK(zeCommandListClose(w->l0.cl));
    ZCK(zeCommandQueueExecuteCommandLists(w->l0.cq, 1, &w->l0.cl, nullptr));
    ZCK(zeCommandQueueSynchronize(w->l0.cq, UINT64_MAX));
    ZCK(zeCommandListReset(w->l0.cl));
}

// ===========================================================================
// 状态管理
// ===========================================================================
void NpuWindow::reset_state() {
    for (size_t i = 0; i < st_ptr.size(); i++)
        memset(st_ptr[i], 0, st_bytes[i]);
}

void NpuWindow::rollback_snapshot(const float* att, const float* wkv,
                                  const float* ffn, int col) {
    for (int l = 0; l < L; l++) {
        if (name2idx[l][0] >= 0)
            memcpy(st_ptr[name2idx[l][0]], att + ((size_t)l * n + col) * D,
                   (size_t)D * 4);
        if (name2idx[l][1] >= 0)
            memcpy(st_ptr[name2idx[l][1]], wkv + ((size_t)l * n + col) * H * hs * hs,
                   (size_t)H * hs * hs * 4);
        if (name2idx[l][2] >= 0)
            memcpy(st_ptr[name2idx[l][2]], ffn + ((size_t)l * n + col) * D,
                   (size_t)D * 4);
    }
}

void NpuWindow::rollback_to(int col) {
    rollback_snapshot(
        static_cast<const float*>(req.get_output_tensor(1).data()),
        static_cast<const float*>(req.get_output_tensor(2).data()),
        static_cast<const float*>(req.get_output_tensor(3).data()),
        col);
}

void NpuWindow::fill_emb_slot(int slot, const EmbeddingTable& emb, int64_t tok) {
    float* row = static_cast<float*>(req.get_input_tensor(0).data()) + (size_t)slot * D;
    emb.fill_row(row, tok);
}

}  // namespace zwin
