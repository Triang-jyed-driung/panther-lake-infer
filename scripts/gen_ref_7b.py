"""
Generate reference logits for RWKV-7 7B using RWKV PIP API with strategy='cpu fp16'.
Runs 10 autoregressive steps starting from a seed token, saving each step's logits.

Usage: python scripts/gen_ref_7b.py
Output: /tmp/rwkv_expected_t{0..9}.bin
"""
import sys, os, numpy as np

os.environ['RWKV_JIT_ON'] = '1'
os.environ['RWKV_CUDA_ON'] = '0'
os.environ['RWKV_V7_ON'] = '1'
os.environ['RESCALE_LAYER'] = '6'

# Add the model path (RWKV v7 appends .pth automatically, so strip it if present)
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else '/D/rwkv7-g1i_preview3260-7.2b-20260716-ctx12288'
if MODEL_PATH.endswith('.pth'):
    MODEL_PATH = MODEL_PATH[:-4]

print(f"Loading model with strategy='cpu fp16'...")
from rwkv.model import RWKV

model = RWKV(MODEL_PATH, strategy='cpu fp16')
print("Model loaded.")

# Start token (same as correctness test)
token = 12605
state = None

for step in range(10):
    logits, state = model.forward([token], state, full_output=False)
    # logits shape: [1, VOCAB] or [VOCAB]
    logits = logits.squeeze().cpu().numpy().astype(np.float32)
    
    # Save
    path = f'/tmp/rwkv_expected_t{step}.bin'
    logits.tofile(path)
    print(f"Step {step}: token_in={token}  argmax={np.argmax(logits)}  "
          f"max={logits.max():.3f}  sum={logits.sum():.3f}  -> {path}")
    
    # Next token via argmax
    token = int(np.argmax(logits))

print("Done.")
