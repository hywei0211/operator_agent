#!/bin/bash
#SBATCH --job-name=verify_test
#SBATCH --output=logs/verify_test-%j.out
#SBATCH --error=logs/verify_test-%j.err
#SBATCH --partition=fnlp-4090
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

cd /remote-home1/hywei/operator_agent_2
export PATH=/usr/local/cuda/bin:$PATH

python3 << 'PYEOF'
import sys, ctypes
sys.path.insert(0, '.')
import torch

# 直接测试 silu_backward kernel
so_path = './output/full_agent/silu_backward.so'
lib = ctypes.CDLL(so_path)
fn = lib.launch_kernel
fn.restype = None
fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

N, H = 64, 1024
x = torch.randn(N, H, dtype=torch.float16, device='cuda')
go = torch.randn(N, H, dtype=torch.float16, device='cuda') * 1.0
gi_fp32 = torch.empty(N*H, dtype=torch.float32, device='cuda')

fn(go.data_ptr(), x.reshape(-1).data_ptr(), gi_fp32.data_ptr(), N*H)
torch.cuda.synchronize()

print(f'grad_in_fp32 nan={gi_fp32.isnan().any().item()}, shape={gi_fp32.shape}')
print(f'first 5: {gi_fp32[:5].tolist()}')

# reference
sig = torch.sigmoid(x.reshape(-1).float())
ref = (go.reshape(-1).float() * sig * (1.0 + x.reshape(-1).float() * (1.0 - sig)))
rel_err = ((gi_fp32 - ref).abs() / (ref.abs() + 1e-6)).max().item()
print(f'rel_err={rel_err:.6f}')
PYEOF
