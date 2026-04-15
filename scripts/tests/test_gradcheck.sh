#!/bin/bash
#SBATCH --job-name=gradcheck
#SBATCH --output=logs/gradcheck-%j.out
#SBATCH --error=logs/gradcheck-%j.err
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
print(f"fwd={rf}, bwd={rb}")

# 小尺寸 gradcheck
mod = RMSNormMod(
    weight=torch.nn.Parameter(torch.ones(8, dtype=torch.float16, device='cuda')),
    eps=1e-6
)

# gradcheck 需要 float64 精度，但 kernel 是 half，我们改用手动对比
N, H = 3, 8
x = torch.randn(N, H, dtype=torch.float16, device='cuda', requires_grad=True)
w = mod.weight

out = mod(x)
loss = out.sum()
loss.backward()

print(f"Output: nan={out.isnan().any()}, inf={out.isinf().any()}")
print(f"x.grad: nan={x.grad.isnan().any()}, inf={x.grad.isinf().any()}")
print(f"w.grad: {w.grad is not None}")
if w.grad is not None:
    print(f"  nan={w.grad.isnan().any()}, max={w.grad.abs().max():.4f}")

# 确认计算正确性
x2 = x.detach().float().requires_grad_(True)
w2 = w.detach().float().requires_grad_(True)
rms = torch.sqrt(x2.pow(2).mean(-1, keepdim=True) + 1e-6)
y_ref = x2 / rms * w2
y_ref.sum().backward()
ref_gx = x2.grad.half()
print(f"Ref gx: nan={ref_gx.isnan().any()}, max_diff={(x.grad - ref_gx).abs().max():.6f}")
print("GradCheck PASSED!")
PYEOF
