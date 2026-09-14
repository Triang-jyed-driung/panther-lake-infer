"""
Convert RWKV-7 .pth (bfloat16) weights to a 16-bit binary for C++ inference.

Memory-efficient version: uses torch.load(mmap=True) to avoid loading the full
model into RAM. Large matrices are converted in chunks. Peak RAM is ~512 MB
even for the 7B model.

Model dimensions are auto-detected from the checkpoint and stored in the
binary header, so the C++ engine can load any model size without recompilation.

Output format (--dtype):
  fp16  (default) — IEEE 754 half precision, magic "RWKV7"
  bf16            — bfloat16 (native .pth dtype), magic "RWKV7B"; keeps the
                    fp32 exponent range, so small-magnitude weights that
                    underflow in fp16 are preserved (higher precision).

Usage: python scripts/convert_weights.py [model.pth] [output.bin] [--dtype fp16|bf16]
  Default: rwkv7-g1d-0.1b-20260129-ctx8192.pth → rwkv7_0.1b_fp16.bin
  Example: python scripts/convert_weights.py /D/rwkv7-7b.pth rwkv7_7b_bf16.bin --dtype bf16
"""
import torch, numpy as np, struct, sys, os, argparse

# ── CLI ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Convert RWKV-7 .pth weights to 16-bit binary')
parser.add_argument('model',  nargs='?', default='rwkv7-g1d-0.1b-20260129-ctx8192.pth',
                    help='input .pth checkpoint (default: rwkv7-g1d-0.1b-20260129-ctx8192.pth)')
parser.add_argument('output', nargs='?', default=None,
                    help='output .bin file (default: rwkv7_0.1b_<dtype>.bin)')
parser.add_argument('--dtype', choices=['fp16', 'bf16'], default='fp16',
                    help='output 16-bit format (default: fp16)')
args = parser.parse_args()

MODEL_PATH = args.model
OUT_DTYPE  = args.dtype
OUT_PATH   = args.output or f'rwkv7_0.1b_{OUT_DTYPE}.bin'
MAGIC      = b'RWKV7' if OUT_DTYPE == 'fp16' else b'RWKV7B'

# ── Tuning ──────────────────────────────────────────────────────────
# Max fp32 elements per chunk. 32M elements = 128 MB.
# Reduce if you have very little RAM.
MAX_CHUNK_ELEMS = 32_000_000


def chunked_convert(f, t, preprocess, dtype, ln0_w=None, ln0_b=None):
    """
    Convert tensor 't' to the selected 16-bit format (fp16 or bf16) and
    write to file 'f'.  Large tensors are processed in chunks to limit RAM.
    """
    t = t.contiguous()
    shape = t.shape
    total = t.numel()
    rank = len(shape)

    def emit(t32):
        """Write a contiguous fp32 torch tensor as the selected format."""
        t32 = t32.contiguous()
        if dtype == 'fp16':
            f.write(t32.numpy().astype(np.float16).tobytes())
        else:  # bf16 — no numpy dtype, write raw 2-byte bf16 bits
            f.write(t32.to(torch.bfloat16).view(torch.int16).numpy().tobytes())

    if total <= MAX_CHUNK_ELEMS:
        # Small enough — convert directly
        if preprocess == 'transpose':
            emit(t.float().t().contiguous())
        elif preprocess == 'ravel':
            emit(t.float().flatten())
        elif preprocess == 'ln0_emb':
            # Single chunk: load full, normalize, convert
            emb = t.float()
            mean = emb.mean(dim=1, keepdim=True)
            var = ((emb - mean) ** 2).mean(dim=1, keepdim=True)
            emb = (emb - mean) / torch.sqrt(var + 1e-5)
            emb = emb * ln0_w + ln0_b
            emit(emb)
        else:
            emit(t.float())
        return

    # ── Chunked path ────────────────────────────────────────────────
    if preprocess == 'ln0_emb':
        # emb: [VOCAB, DIM] — chunk by rows (tokens)
        n_tok = shape[0]
        chunk_rows = max(1, MAX_CHUNK_ELEMS // shape[1])
        for start in range(0, n_tok, chunk_rows):
            end = min(start + chunk_rows, n_tok)
            chunk = t[start:end].float()
            mean = chunk.mean(dim=1, keepdim=True)
            var = ((chunk - mean) ** 2).mean(dim=1, keepdim=True)
            chunk = (chunk - mean) / torch.sqrt(var + 1e-5)
            chunk = chunk * ln0_w + ln0_b
            emit(chunk)
            del chunk, mean, var

    elif preprocess == 'transpose':
        # [M, N] → transpose → [N, M] — chunk by OUTPUT rows (original columns)
        M, N = int(shape[0]), int(shape[1])
        out_rows = N  # output has N rows, each of M elements
        chunk_rows = max(1, MAX_CHUNK_ELEMS // M)  # how many output rows per chunk
        for start in range(0, out_rows, chunk_rows):
            end = min(start + chunk_rows, out_rows)
            # Read columns [start:end] from original → [M, chunk_rows]
            # Transpose → [chunk_rows, M], write as contiguous block
            chunk = t[:, start:end].float().t().contiguous()
            emit(chunk)
            del chunk

    elif preprocess == 'ravel':
        # Flatten in chunks along dim 0
        chunk_rows = max(1, MAX_CHUNK_ELEMS // (total // shape[0]))
        for start in range(0, shape[0], chunk_rows):
            end = min(start + chunk_rows, shape[0])
            chunk = t[start:end].float().flatten()
            emit(chunk)
            del chunk

    else:
        # Plain: chunk along dim 0
        chunk_rows = max(1, MAX_CHUNK_ELEMS // (total // shape[0]))
        for start in range(0, shape[0], chunk_rows):
            end = min(start + chunk_rows, shape[0])
            chunk = t[start:end].float()
            emit(chunk)
            del chunk


def main():
    # ── Open model with memory-mapping (near-zero RAM) ──────────────
    print(f"Loading {MODEL_PATH} (mmap)...")
    z = torch.load(MODEL_PATH, map_location='cpu', mmap=True)

    # ── Auto-detect dimensions from shapes ──────────────────────────
    N_VOCAB, N_EMBD = int(z['emb.weight'].shape[0]), int(z['emb.weight'].shape[1])

    N_LAYER = 0
    while f'blocks.{N_LAYER}.ln1.weight' in z:
        N_LAYER += 1

    # head_size is the LAST dim of r_k (RWKV-7: r_k is [n_heads, head_size])
    HEAD_SIZE = int(z['blocks.0.att.r_k'].shape[-1])
    N_HEAD = N_EMBD // HEAD_SIZE
    FFN_DIM = int(z['blocks.0.ffn.key.weight'].shape[0])

    print(f"Detected: {N_LAYER} layers, DIM={N_EMBD}, N_HEAD={N_HEAD}, "
          f"H_SIZE={HEAD_SIZE}, VOCAB={N_VOCAB}, FFN={FFN_DIM}")

    # ── Build metadata list ─────────────────────────────────────────
    blocks = []
    def add(name, key, pp=None):
        blocks.append((name, key, pp))

    add('emb.weight', 'emb.weight', 'ln0_emb')
    add('head.weight', 'head.weight', 'transpose')
    add('ln_out.weight', 'ln_out.weight')
    add('ln_out.bias', 'ln_out.bias')

    for i in range(N_LAYER):
        a = f'blocks.{i}.att.'; f_ = f'blocks.{i}.ffn.'
        add(f'L{i}.ln1.w', f'blocks.{i}.ln1.weight')
        add(f'L{i}.ln1.b', f'blocks.{i}.ln1.bias')
        add(f'L{i}.ln2.w', f'blocks.{i}.ln2.weight')
        add(f'L{i}.ln2.b', f'blocks.{i}.ln2.bias')
        for s in ['x_r','x_w','x_k','x_v','x_a','x_g']: add(f'L{i}.att.{s}', f'{a}{s}')
        add(f'L{i}.att.w0', f'{a}w0'); add(f'L{i}.att.w1', f'{a}w1'); add(f'L{i}.att.w2', f'{a}w2')
        add(f'L{i}.att.a0', f'{a}a0'); add(f'L{i}.att.a1', f'{a}a1'); add(f'L{i}.att.a2', f'{a}a2')
        add(f'L{i}.att.v0', f'{a}v0'); add(f'L{i}.att.v1', f'{a}v1'); add(f'L{i}.att.v2', f'{a}v2')
        add(f'L{i}.att.g1', f'{a}g1'); add(f'L{i}.att.g2', f'{a}g2')
        add(f'L{i}.att.k_k', f'{a}k_k'); add(f'L{i}.att.k_a', f'{a}k_a')
        add(f'L{i}.att.r_k', f'{a}r_k', 'ravel')
        add(f'L{i}.att.R', f'{a}receptance.weight', 'transpose')
        add(f'L{i}.att.K', f'{a}key.weight', 'transpose')
        add(f'L{i}.att.Vmat', f'{a}value.weight', 'transpose')
        add(f'L{i}.att.O', f'{a}output.weight', 'transpose')
        add(f'L{i}.att.ln_x.w', f'{a}ln_x.weight'); add(f'L{i}.att.ln_x.b', f'{a}ln_x.bias')
        add(f'L{i}.ffn.x_k', f'{f_}x_k')
        add(f'L{i}.ffn.K', f'{f_}key.weight', 'transpose')
        add(f'L{i}.ffn.V', f'{f_}value.weight', 'transpose')

    # ── Compute sizes ───────────────────────────────────────────────
    sizes = []
    for name, key, pp in blocks:
        t = z[key]
        if pp == 'transpose':
            n = int(t.shape[0]) * int(t.shape[1])
        else:
            n = t.numel()
        sizes.append(n)

    # ── Write header + offset table + name table ────────────────────
    print(f"Writing {OUT_PATH}  (dtype={OUT_DTYPE}, magic={MAGIC.decode()})...")
    with open(OUT_PATH, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('IIIIIII', N_LAYER, N_EMBD, N_HEAD, HEAD_SIZE,
                            N_VOCAB, FFN_DIM, len(blocks)))

        name_table_ofs = 8 + 28 + len(blocks) * 16
        for name, _, _ in blocks:
            name_table_ofs += 2 + len(name)
        data_start = (name_table_ofs + 63) & ~63

        off = data_start
        for n in sizes:
            f.write(struct.pack('QQ', off, n))
            off += n * 2

        for name, _, _ in blocks:
            nb = name.encode('utf-8')
            f.write(struct.pack('H', len(nb)))
            f.write(nb)

        while f.tell() < data_start:
            f.write(b'\x00')

        # ── Preload ln0 params ──────────────────────────────────────
        ln0_w = z['blocks.0.ln0.weight'].float()
        ln0_b = z['blocks.0.ln0.bias'].float()

        # ── Stream-write each block ─────────────────────────────────
        total_kb = 0
        for idx, (name, key, pp) in enumerate(blocks):
            t = z[key]
            kb = sizes[idx] * 2 // 1024
            print(f"  [{idx+1:4d}/{len(blocks)}] {name:30s} {kb:8d} KB")
            chunked_convert(f, t, pp, OUT_DTYPE, ln0_w, ln0_b)
            total_kb += kb

    print(f"\nTotal: {total_kb / 1024:.1f} MB ({len(blocks)} blocks)")
    print("Done!")


if __name__ == '__main__':
    main()
