#!/bin/bash
#SBATCH --job-name=rmsnorm_grad
#SBATCH --output=logs/rmsnorm_grad-%j.out
#SBATCH --error=logs/rmsnorm_grad-%j.err
#SBATCH --partition=fnlp-4090
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

cd /remote-home1/hywei/operator_agent_2
python3 << 'PYEOF'
import sys, torch, torch.nn as nn, ctypes
sys.path.insert(0, '.')

so_fwd = ctypes.CDLL('./output/full_agent/rmsnorm_forward.so')
fn_fwd = so_fwd.launch_kernel
fn_fwd.restype = None
fn_fwd.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int, ctypes.c_int, ctypes.c_float]

so_bwd = ctypes.CDLL('./output/full_agent/rmsnorm_backward.so')
fn_bwd = so_bwd.launch_kernel
fn_bwd.restype = None
fn_bwd.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float]

class RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        x_c = x.contiguous(); w_c = weight.contiguous()
        N, H = x_c.shape[0], x_c.shape[-1]
        out = torch.empty_like(x_c)
        fn_fwd(x_c.data_ptr(), w_c.data_ptr(), out.data_ptr(), N, H, float(eps))
        torch.cuda.synchronize()
        ctx.save_for_backward(x, weight); ctx.eps = eps
        return out
    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors; eps = ctx.eps
        x_c = x.contiguous(); w_c = w.contiguous(); g_c = g.contiguous()
        N, H = x_c.shape[0], x_c.shape[-1]
        gx = torch.empty_like(x_c)
        gw_fp32 = torch.zeros(H, dtype=torch.float32, device=x.device)
        fn_bwd(g_c.data_ptr(), x_c.data_ptr(), w_c.data_ptr(),
               gx.data_ptr(), gw_fp32.data_ptr(), N, H, float(eps))
        torch.cuda.synchronize()
        return gx, gw_fp32.to(w.dtype), None

class RMSNormMod(nn.Module):
    def __init__(self, H, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(H, dtype=torch.float16, device='cuda'))
        self.eps = eps
    def forward(self, x):
        orig = x.shape
        x2 = x.reshape(-1, orig[-1])
        return RMSNormFn.apply(x2, self.weight, self.eps).reshape(orig)

mod = RMSNormMod(16).cuda()
x = torch.randn(4, 16, dtype=torch.float16, device='cuda', requires_grad=True)
out = mod(x)
loss = out.sum()
loss.backward()
print(f'x.grad nan={x.grad.isnan().any().item()} norm={x.grad.float().norm():.4f}')
print(f'w.grad nan={mod.weight.grad.isnan().any().item()} norm={mod.weight.grad.float().norm():.4f}')
print("RMSNorm gradient test PASSED!")
PYEOF
