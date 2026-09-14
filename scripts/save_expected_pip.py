"""Generate reference logits using official RWKV PIP (black-box)."""
import os, numpy as np
os.environ['RWKV_V7_ON'] = '1'
os.environ['RWKV_CUDA_ON'] = '0'
os.environ['RWKV_JIT_ON'] = '0'
import torch
from rwkv.model import RWKV

MODEL_PATH = "/D/py/rwkv7-g1d-0.1b-20260129-ctx8192"
OUT_DIR = "/tmp"

print("Loading model via RWKV PIP (cpu fp32)...")
model = RWKV(MODEL_PATH, "cpu fp32")
state = model.generate_zero_state()

token = 12605
for step in range(10):
    logits, state = model.forward([token], state)
    arr = logits.cpu().float().numpy().flatten()
    next_token = int(np.argmax(arr))
    
    path = f'{OUT_DIR}/rwkv_expected_t{step}.bin'
    arr.astype(np.float32).tofile(path)
    
    print(f'Step {step}: token_in={token:>6}  argmax={next_token:>6}  sum={arr.sum():.2f}  [0:8]={arr[:8]}')
    token = next_token

print("Done. Reference files written to /tmp/rwkv_expected_t{0..9}.bin")
