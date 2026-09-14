"""Generate reference logits for 10-token autoregressive decode starting from token 33."""
import numpy as np, struct, os

BIN = 'rwkv7_0.1b_fp16.bin'
OUT_DIR = '/tmp'
C, H, N, FF = 768, 12, 64, 3072
VOCAB = 65536

with open(BIN, 'rb') as f:
    f.read(5)
    hdr = struct.unpack('IIIIIII', f.read(28))
    nb = hdr[6]
    offs, szs = [], []
    for _ in range(nb):
        o, s = struct.unpack('QQ', f.read(16))
        offs.append(o); szs.append(s)
    nms = []
    for _ in range(nb):
        l = struct.unpack('H', f.read(2))[0]
        nms.append(f.read(l).decode())
    W = {}
    for i, nm in enumerate(nms):
        f.seek(offs[i])
        raw = np.frombuffer(f.read(szs[i] * 2), dtype=np.float16).astype(np.float32)
        W[nm] = raw


def sig(x):
    return 1 / (1 + np.exp(np.clip(-x, -20, 20)))


def ln(x, w, b):
    m = x.mean()
    v_ = ((x - m) ** 2).mean()
    return (x - m) / np.sqrt(v_ + 1e-5) * w + b


def forward_one(x, axp, ast, fxp, vf):
    for l in range(12):
        xx = ln(x, W[f'L{l}.ln1.w'], W[f'L{l}.ln1.b'])
        t1 = axp[l] - xx
        axp[l] = xx.copy()
        r = xx + t1 * W[f'L{l}.att.x_r']
        r = r @ W[f'L{l}.att.R'].reshape(C, C)
        w_ = xx + t1 * W[f'L{l}.att.x_w']
        w_ = np.tanh(w_ @ W[f'L{l}.att.w1'].reshape(C, 64)) @ W[f'L{l}.att.w2'].reshape(64, C)
        k_ = xx + t1 * W[f'L{l}.att.x_k']
        k_ = k_ @ W[f'L{l}.att.K'].reshape(C, C)
        v_ = xx + t1 * W[f'L{l}.att.x_v']
        xv = v_.copy()
        v_ = v_ @ W[f'L{l}.att.Vmat'].reshape(C, C)
        a_ = xx + t1 * W[f'L{l}.att.x_a']
        a_ = sig(W[f'L{l}.att.a0'] + (a_ @ W[f'L{l}.att.a1'].reshape(C, 64)) @ W[f'L{l}.att.a2'].reshape(64, C))
        g_ = xx + t1 * W[f'L{l}.att.x_g']
        g_ = sig(g_ @ W[f'L{l}.att.g1'].reshape(C, 128)) @ W[f'L{l}.att.g2'].reshape(128, C)
        kk = np.zeros(C)
        for h in range(H):
            s, e = h * N, (h + 1) * N
            vals = k_[s:e] * W[f'L{l}.att.k_k'][s:e]
            kk[s:e] = vals / np.sqrt((vals ** 2).sum())
        k_ = k_ * (1 + (a_ - 1) * W[f'L{l}.att.k_a'])
        if l == 0:
            vf = v_.copy()
        else:
            sv_ = sig(W[f'L{l}.att.v0'] + (xv @ W[f'L{l}.att.v1'].reshape(C, 32)) @ W[f'L{l}.att.v2'].reshape(32, C))
            v_ = v_ + (vf - v_) * sv_
        xxw = np.zeros(C)
        for h in range(H):
            S = ast[l][h]
            off = h * N
            wv = np.exp(-0.606531 * sig(W[f'L{l}.att.w0'][off:off + N] + w_[off:off + N]))
            kka = kk[off:off + N] * a_[off:off + N]
            for i in range(N):
                Si = S[i]
                sa = (Si * (-kk[off:off + N])).sum()
                for j in range(N):
                    Si[j] = Si[j] * wv[j] + v_[off + i] * k_[off + j] + sa * kka[j]
                xxw[off + i] = (Si * r[off:off + N]).sum()
        xxg = np.zeros(C)
        for h in range(H):
            xh = xxw[h * N:(h + 1) * N]
            m = xh.mean()
            v__ = ((xh - m) ** 2).mean()
            xxg[h * N:(h + 1) * N] = (xh - m) / np.sqrt(v__ + 64e-5) * W[f'L{l}.att.ln_x.w'][h * N:(h + 1) * N] + W[f'L{l}.att.ln_x.b'][h * N:(h + 1) * N]
        for h in range(H):
            bonus = (r[h * N:(h + 1) * N] * k_[h * N:(h + 1) * N] * W[f'L{l}.att.r_k'][h * N:(h + 1) * N]).sum()
            xxg[h * N:(h + 1) * N] += bonus * v_[h * N:(h + 1) * N]
        xo = (xxg * g_) @ W[f'L{l}.att.O'].reshape(C, C)
        x = x + xo
        xx2 = ln(x, W[f'L{l}.ln2.w'], W[f'L{l}.ln2.b'])
        t1 = fxp[l] - xx2
        fxp[l] = xx2.copy()
        k_ = xx2 + t1 * W[f'L{l}.ffn.x_k']
        kr = np.maximum(0, k_ @ W[f'L{l}.ffn.K'].reshape(C, FF)) ** 2
        xf = kr @ W[f'L{l}.ffn.V'].reshape(FF, C)
        x = x + xf
    xo = ln(x, W['ln_out.weight'], W['ln_out.bias'])
    logits = xo @ W['head.weight'].reshape(C, VOCAB)
    return logits.astype(np.float32), x, axp, ast, fxp, vf


# ── Autoregressive decode: start at 33, sample argmax for 10 steps ──
axp = [np.zeros(C) for _ in range(12)]
ast = [np.zeros((H, N, N)) for _ in range(12)]
fxp = [np.zeros(C) for _ in range(12)]
vf = np.zeros(C)

token = 33
for step in range(10):
    x = W['emb.weight'].reshape(VOCAB, C)[token].copy()
    logits, x, axp, ast, fxp, vf = forward_one(x, axp, ast, fxp, vf)
    next_token = int(np.argmax(logits))
    path = os.path.join(OUT_DIR, f'rwkv_expected_t{step}.bin')
    logits.tofile(path)
    print(f'Step {step}: token_in={token}  argmax={next_token}  sum={logits.sum():.2f}  [0:8]={logits[:8]}')
    token = next_token
