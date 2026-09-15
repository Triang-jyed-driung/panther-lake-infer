"""
window_v5d.py — 有状态窗口图，词嵌入直读版（去掉图内 512MiB emb 常量）
================================================================
与 window_v5c.py 的唯一区别：输入不是 token id，而是融合 ln0 之后的词嵌入行，
图内不再有 [V,D] fp16 词嵌入常量（7B 为 65536×4096 = 512MiB），blob 与设备
内存各省 512MiB。数值与 v5c 逐位一致：宿主侧用导出的融合嵌入表（fp16）查行，
以 f32 喂入——等价于图内 gather(f16)+convert(f32)。

接口（ReadValue/Assign 有状态，与 v5c 相同）：
  输入：
    in[0]        emb_x [n,D] f32 —— 融合 ln0 后的词嵌入行（fp16 值的 f32 表示）
  输出：
    out[0] logits [n,V] f16
    out[1] att  [L,n,D] f32
    out[2] wkv  [L,n,H,hs,hs] f32
    out[3] ffn  [L,n,D] f32
  状态变量：每层 {L}.xp / {L}.wkv / {L}.fp，共 3L 个（query_state 管理）。

构建时把融合嵌入表导出为 export/emb_{tag}.f16.bin（[V,D] fp16 连续，512MiB），
meta json 记录路径，供宿主侧查行。

用法:
  python window_v5d.py [pth] [n]    # 构建 + 序列化 xml/bin + 导出 emb 表
"""
import os
import sys
import gc
import json
import time
import numpy as np
import openvino as ov
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(HERE, "export")
CACHE_DIR = os.path.join(HERE, "npu_cache")
DEFAULT_PTH = "/D/g1d_0.1b.pth"
DEFAULT_N = 4

O = ov.opset13
i32, f16, f32 = ov.Type.i32, ov.Type.f16, ov.Type.f32
def MM(a, b): return O.matmul(a, b, False, False)
def UD(a, n): return O.unsqueeze(a, n)
def KI(data):
    if isinstance(data, (int, float)):
        return O.constant(np.array([data], np.int32))
    return O.constant(np.asarray(data).astype(np.int32))
def KC(data):
    if isinstance(data, (int, float)):
        return O.constant(np.array([data], np.float16))
    return O.constant(np.asarray(data).astype(np.float16))
def KC32(data):
    if isinstance(data, (int, float)):
        return O.constant(np.array([data], np.float32))
    return O.constant(np.asarray(data).astype(np.float32))
def TO16(x): return O.convert(x, f16)
def TO32(x): return O.convert(x, f32)
def KI64(data): return O.constant(np.asarray(data, np.int64))
def LN(x, w, b, groups=1, eps=1e-5): return O.squeeze(O.group_normalization(
    UD(x, -1), w, b, num_groups=groups, epsilon=np.float32(eps)), [-1])
def RS(x, s): return O.reshape(x, KI64(s), False)
def ROW(x, t): return O.gather(x, O.constant(np.int64(t)), KI64(0))
def STACK(nodes, axis=0): return O.concat([UD(s, axis) for s in nodes], axis)
def MV16(x32, W16): return TO32(MM(TO16(x32), W16))
def FAM(a, b, c): return O.add(a, O.multiply(b, c))

def wkv_window(S, r32, dm1, k32, v32, kkn32, kka32, n):
    """n 步 wkv 递推，全 f32（含内部小 matmul）。参数为逐 t 的 [H,hs] f32 节点。"""
    ys, snaps = [], []
    for t in range(n):
        Sd = O.multiply(S, UD(dm1[t], -2))
        kv = MM(UD(v32[t], -1), UD(k32[t], -2))
        sab = MM(MM(S, UD(kkn32[t], -1)), UD(kka32[t], -2))
        # 。。。。
        Sn = O.add(O.add(Sd, kv), sab)
        ys.append(O.squeeze(MM(Sn, UD(r32[t], -1)), -1))
        snaps.append(Sn)
        S = Sn
    return ys, snaps


def build(pth: str, n: int, emb_out: str):
    raw = torch.load(pth, map_location="cpu", mmap=True)
    n_layer = 0
    while f"blocks.{n_layer}.ln1.weight" in raw:
        n_layer += 1
    D = raw["emb.weight"].shape[1]
    V = raw["emb.weight"].shape[0]
    H, hs = raw["blocks.0.att.r_k"].shape
    print(f"L={n_layer} C={D} H={H} N={hs} V={V}", flush=True)

    w = {}
    def W(_x): nonlocal w; return KC(w[_x])
    def WL(_x): nonlocal w; return KC32(w[_x])

    for k, v in raw.items():
        v = v.squeeze().to(torch.float16)
        if any(s in k for s in ["key.weight", "value.weight", "receptance.weight",
                                "output.weight", "head.weight"]):
            v = v.t().contiguous()
        if k.endswith("att.r_k"):
            v = v.flatten().contiguous()
        w[k] = v.numpy()
    emb_raw = raw["emb.weight"].squeeze().float().numpy()
    lw0 = raw["blocks.0.ln0.weight"].squeeze().float().numpy()
    lb0 = raw["blocks.0.ln0.bias"].squeeze().float().numpy()
    m_ = emb_raw.mean(-1, keepdims=True)
    v_ = emb_raw.var(-1, keepdims=True)
    emb = (((emb_raw - m_) / np.sqrt(v_ + 1e-5)) * lw0 + lb0).astype(np.float16)
    del raw
    gc.collect()

    # 融合嵌入表导出（[V,D] fp16 连续，宿主查行用；7B 512MiB）
    if not os.path.exists(emb_out):
        emb.tofile(emb_out)
        print(f"emb table -> {emb_out} ({os.path.getsize(emb_out)/2**20:.0f}MiB)", flush=True)
    else:
        print(f"emb table exists -> {emb_out}", flush=True)
    del emb_raw, emb
    gc.collect()

    print("Building...", flush=True)
    t0 = time.time()
    emb_in = O.parameter([n, D], f32, "emb_x")
    # T = O.parameter([1], i32, "elapsed_tokens")
    x = emb_in                                                    # 残差流 f32

    att_snaps, wkv_snaps, ffn_snaps = [], [], []
    sinks = []
    vf = None

    for i in range(n_layer):
        vid = f"L{i}"
        p = f"blocks.{i}."
        a = f"{p}att."
        f = f"{p}ffn."

        rv_xp = O.read_value(f"{vid}.xp", f32, [D])
        rv_ws = O.read_value(f"{vid}.wkv", f32, [H, hs, hs])
        rv_fp = O.read_value(f"{vid}.fp", f32, [D])

        # matmul 权重（f16）
        Rw = W(a + "receptance.weight"); Kw = W(a + "key.weight")
        Vw = W(a + "value.weight"); Ow = W(a + "output.weight")
        w1 = W(a + "w1"); w2 = W(a + "w2")
        a1 = W(a + "a1"); a2 = W(a + "a2")
        g1 = W(a + "g1"); g2 = W(a + "g2")
        v1 = W(a + "v1"); v2 = W(a + "v2")
        fK = W(f + "key.weight"); fV = W(f + "value.weight")
        hd = W("head.weight")
        # 逐元素 / norm 参数（f32）
        l1w = WL(p + "ln1.weight"); l1b = WL(p + "ln1.bias")
        l2w = WL(p + "ln2.weight"); l2b = WL(p + "ln2.bias")
        xr_w = WL(a + "x_r"); xw_w = WL(a + "x_w"); xk_w = WL(a + "x_k")
        xv_w = WL(a + "x_v"); xa_w = WL(a + "x_a"); xg_w = WL(a + "x_g")
        w0_32 = WL(a + "w0"); a0_32 = WL(a + "a0")
        v0_32 = WL(a + "v0")
        kkc32 = WL(a + "k_k"); kac32 = WL(a + "k_a")
        rk32 = WL(a + "r_k")
        lxw = WL(a + "ln_x.weight"); lxb = WL(a + "ln_x.bias")
        fxk = WL(f + "x_k")
        lnw = WL("ln_out.weight"); lnb = WL("ln_out.bias")
        
        
        # LN1 + token-shift（f32）
        xx = LN(x, l1w, l1b)
        att_snaps.append(xx)
        prev = O.concat([UD(rv_xp, 0),
                         O.slice(xx, KI64([0, 0]), KI64([n - 1, D]),
                                 KI64([1, 1]), KI64([0, 1]))], 0)
        dx = O.subtract(prev, xx)

        xr = FAM(xx, dx, xr_w)
        xw = FAM(xx, dx, xw_w)
        xk = FAM(xx, dx, xk_w)
        xv = FAM(xx, dx, xv_w)
        xa = FAM(xx, dx, xa_w)
        xg = FAM(xx, dx, xg_w)

        # GEMM（f16）+ 逐元素（f32）
        r32 = MV16(xr, Rw)
        k32 = MV16(xk, Kw)
        v32 = MV16(xv, Vw)
        ws32 = O.add(w0_32, MV16(O.tanh(MV16(xw, w1)), w2))

        ac = KC32(-0.6065306597126334)
        ONE = KC32(1.0)
        # TN = UD(KI(range(n)), -1) # (n, 1)
        # RO1 = KI(40503)
        # ts = O.add(TN, UD(T, -1))

        fn = O.exp(O.divide(ac, O.add(ONE, O.exp(O.negative(ws32)))))
        # === 分支2: x < -3 近似 ===
        # y = exp(-8)*exp(x+8) = exp(x)
        y = O.multiply(KC32(0.00033546262790251185), O.exp(O.add(ws32, KC32(8.0)))) # exp(-8)
        # F = a*y*(1+b*y)
        fa = O.add(ONE, O.multiply(O.multiply(ac, y), O.subtract(ONE, O.multiply(KC32(1.3032653298563167), y))))

        dm32 = O.select(O.greater_equal(ws32, KC32(-3.0)), fn, fa)


        a32 = O.sigmoid(O.add(a0_32, MV16(MV16(xa, a1), a2)))
        g32 = MV16(O.sigmoid(MV16(xg, g1)), g2)

        # kk / k2（f32）
        kh = RS(k32, [n, H, hs])
        kkc = RS(kkc32, [H, hs])
        kkh = O.multiply(kh, kkc)
        kki = O.divide(KC32(1.0), O.add(O.sqrt(O.reduce_sum(
            O.multiply(kkh, kkh), [2], True)), KC32(1e-12)))
        kk2 = O.multiply(kkh, kki)                            # [n,H,hs]（正号）
        k2 = O.multiply(kh, O.add(KC32(1.0), O.multiply(
            O.subtract(RS(a32, [n, H, hs]), KC32(1.0)), RS(kac32, [H, hs]))))
        kkac = O.multiply(kk2, RS(a32, [n, H, hs]))

        if i == 0:
            vo = v32
            nvf = v32
        else:
            vg = O.sigmoid(O.add(v0_32, MV16(MV16(xv, v1), v2)))
            vo = O.add(v32, O.multiply(O.subtract(vf, v32), vg))
            nvf = vf

        # 逐 t wkv 递推（f32）
        ys, snaps = wkv_window(rv_ws,
                               [RS(ROW(r32, t), [H, hs]) for t in range(n)],
                               [RS(ROW(dm32, t), [H, hs]) for t in range(n)],
                               [ROW(k2, t) for t in range(n)],
                               [RS(ROW(vo, t), [H, hs]) for t in range(n)],
                               [O.negative(ROW(kk2, t)) for t in range(n)],
                               [ROW(kkac, t) for t in range(n)], n)
        wkv_snaps.append(STACK(snaps))                        # [n,H,hs,hs] f32

        y_all = STACK(ys)                                     # [n,H,hs] f32
        yn = LN(RS(y_all, [n, D]), lxw, lxb, groups=H, eps=64e-5)
        boost = O.multiply(O.reduce_sum(O.multiply(O.multiply(
            RS(r32, [n, H, hs]), RS(k2, [n, H, hs])), RS(rk32, [H, hs])), [2], True),
            RS(vo, [n, H, hs]))                               # [n,H,1]
        yg = O.multiply(O.add(yn, RS(boost, [n, D])), g32)    # [n,D]
        x = O.add(x, MV16(yg, Ow))

        # 状态更新（窗口最后一行，f32）+ 快照
        sinks.append(O.assign(ROW(xx, n - 1), f"{vid}.xp"))
        sinks.append(O.assign(snaps[-1], f"{vid}.wkv"))

        # ffn（token-shift 窗口化，f32）
        xx2 = LN(x, l2w, l2b)
        ffn_snaps.append(xx2)
        prev2 = O.concat([UD(rv_fp, 0),
                          O.slice(xx2, KI64([0, 0]), KI64([n - 1, D]),
                                  KI64([1, 1]), KI64([0, 1]))], 0)
        dx2 = O.subtract(prev2, xx2)
        kf32 = O.add(xx2, O.multiply(dx2, fxk))
        kh_ = O.relu(MV16(kf32, fK))
        kf32 = O.multiply(kh_, kh_)                           # relu²
        x = O.add(x, MV16(kf32, fV))
        sinks.append(O.assign(ROW(xx2, n - 1), f"{vid}.fp"))
        vf = nvf

    n_out = LN(x, lnw, lnb)
    logits = MM(TO16(n_out), hd)           # [n,V] f16

    model = ov.Model(
        [O.result(logits),
         O.result(STACK(att_snaps)),                          # [L,n,D] f32
         O.result(STACK(wkv_snaps)),                          # [L,n,H,hs,hs] f32
         O.result(STACK(ffn_snaps))],                         # [L,n,D] f32
        [s.output(0) for s in sinks],
        [emb_in])
    print(f"  graph: {time.time()-t0:.1f}s, vars={len(model.get_variables())}", flush=True)
    return model, dict(V=int(V), D=int(D), H=int(H), hs=int(hs), L=int(n_layer))


def compile_for(core, xml, device):
    """GPU 必须 inference_precision=f32（否则逐元素走 f16 fast-math，ReLU 等都不精确，
    见 ops_crossval.py）。NPU 无精度选项（f16 运算是硬件特性）。"""
    props = {"CACHE_DIR": CACHE_DIR}
    if device == "GPU":
        from openvino.properties import hint as hints
        props[hints.inference_precision] = ov.Type.f32
    return core.compile_model(xml, device, props)


def main():
    pth = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PTH
    n = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N
    os.makedirs(EXPORT_DIR, exist_ok=True)

    tag = os.path.basename(pth).replace(".pth", "")
    stem = os.path.join(EXPORT_DIR, f"window_v6_{tag}_n{n}")
    xml = stem + ".xml"
    emb_out = os.path.join(EXPORT_DIR, f"emb_{tag}.f16.bin")
    model, dm = build(pth, n, emb_out)
    ov.serialize(model, xml)
    json.dump(dict(dm=dm, n=n, pth=pth, graph="v6-emb", emb=emb_out),
              open(stem + ".json", "w"))
    print(f"serialized -> {xml}", flush=True)
    del model
    gc.collect()


if __name__ == "__main__":
    main()
