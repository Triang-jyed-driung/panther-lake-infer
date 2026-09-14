"""Test PIP API directly."""
import os, numpy as np
os.environ['RWKV_V7_ON'] = '1'
os.environ['RWKV_CUDA_ON'] = '0'
os.environ['RWKV_JIT_ON'] = '0'
import torch
from rwkv.model import RWKV

m = RWKV('/D/py/rwkv7-g1d-0.1b-20260129-ctx8192', 'cpu fp32')
state = m.generate_zero_state()

logits, state = m.forward([33], state)
arr = logits.cpu().numpy().flatten()
print('shape:', logits.shape)
print('[0:8]:', arr[:8])
print('argmax:', int(np.argmax(arr)))
print('state type:', type(state), 'len:', len(state))
