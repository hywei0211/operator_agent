#!/bin/bash
#SBATCH --job-name=rmsnorm_h128
#SBATCH --output=logs/rmsnorm_h128-%j.out
#SBATCH --error=logs/rmsnorm_h128-%j.err
#SBATCH --partition=fnlp-4090
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

cd /remote-home1/hywei/operator_agent_2
python3 << 'PYEOF'
import sys, torch, ctypes
sys.path.insert(0, '.')

so_bwd = ctypes.CDLL('./output/full_agent/rmsnorm_backward.so')
fn_bwd = so_bwd.launch_kernel
fn_bwd.restype = None
fn_bwd.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float]

for H in [128, 1024]:
    for N in [1, 4, 64, 256]:
        x = torch.randn(N, H, dtype=torch.float16, device='cuda')
        w = torch.ones(H, dtype=torch.float16, device='cuda')
        go = torch.ones(N, H, dtype=torch.float16, device='cuda') * 0.01
        gx = torch.zeros(N, H, dtype=torch.float16, device='cuda')
        gw = torch.zeros(H, dtype=torch.float32, device='cuda')
        fn_bwd(go.data_ptr(), x.data_ptr(), w.data_ptr(), gx.data_ptr(), gw.data_ptr(), N, H, 1e-6)
        torch.cuda.synchronize()
        nan_x = gx.isnan().any().item()
        nan_w = gw.isnan().any().item()
        print(f"H={H}, N={N}: gx_nan={nan_x}, gw_nan={nan_w}, gx_norm={gx.float().norm():.4f}")
PYEOF
