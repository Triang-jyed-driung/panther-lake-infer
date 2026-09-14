"""Autoregressive decode starting from 12605 using RWKV PIP + tokenizer."""
import os, numpy as np
os.environ['RWKV_V7_ON'] = '1'
os.environ['RWKV_CUDA_ON'] = '0'
os.environ['RWKV_JIT_ON'] = '0'
import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE

MODEL_PATH = "/D/py/rwkv7-g1d-0.1b-20260129-ctx8192"

model = RWKV(MODEL_PATH, "cpu fp32")
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

state = model.generate_zero_state()
token = 12605
print(f"Start token: {token} -> {repr(pipeline.decode([token]))}")

for step in range(10):
    logits, state = model.forward([token], state)
    arr = logits.cpu().float().numpy().flatten()
    next_token = int(np.argmax(arr))
    print(f"  step {step}: in={token:>6} out={next_token:>6}  ({repr(pipeline.decode([next_token]))})")
    token = next_token

# Decode the full sequence
print()
all_tokens = [12605]  # we'd need to collect them, let's re-run
