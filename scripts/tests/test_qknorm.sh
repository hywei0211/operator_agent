#!/bin/bash
#SBATCH --job-name=qknorm
#SBATCH --output=logs/qknorm-%j.out
#SBATCH --error=logs/qknorm-%j.err
#SBATCH --partition=fnlp-4090
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

cd /remote-home1/hywei/operator_agent_2
python3 << 'PYEOF'
import sys, torch, ctypes
sys.path.insert(0, '.')
from examples.full_agent_lora_train import create_custom_rmsnorm

RMSNormMod, rf, rb = create_custom_rmsnorm(
    './output/full_agent/rmsnorm_forward.so',
    './output/full_agent/rmsnorm_backward.so'
)

# q_norm 维度：[1, 16, 6, 128] -> [96, 128]
# k_norm 维度：[1, 8, 6, 128] -> [48, 128]  (num_kv_heads=8)

w128 = torch.nn.Parameter(torch.ones(128, dtype=torch.float16, device='cuda'))
mod = RMSNormMod(w128, eps=1e-6)

for N in [6, 48, 96, 6*16, 6*16*4]:
    x = torch.randn(1, 16, N//16 if N >= 16 else 1, 128, dtype=torch.float16, device='cuda', requires_grad=True)
    try:
        out = mod(x)
        loss = out.sum()
        loss.backward()
        nan_x = x.grad.isnan().any().item()
        nan_w = w128.grad.isnan().any().item() if w128.grad is not None else None
        print(f"N_total={x.numel()//128} (shape={tuple(x.shape)}): x_nan={nan_x}, w_nan={nan_w}")
        w128.grad = None
        x.grad = None
    except Exception as e:
        print(f"ERROR N={N}: {e}")
PYEOF
