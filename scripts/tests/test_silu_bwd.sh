#!/bin/bash
#SBATCH --job-name=silu_bwd
#SBATCH --output=logs/silu_bwd-%j.out
#SBATCH --error=logs/silu_bwd-%j.err
#SBATCH --partition=fnlp-4090
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

cd /remote-home1/hywei/operator_agent_2
python3 << 'PYEOF'
import sys, torch, ctypes
sys.path.insert(0, '.')

so = ctypes.CDLL('./output/full_agent/silu_backward.so')
fn = so.launch_kernel
fn.restype = None
fn.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]

# 测试 SiLU backward 是否产生 NaN（各种输入范围）
for scale in [0.01, 0.1, 1.0, 10.0]:
    for N in [128, 3072, 196608]:
        x = torch.randn(N, dtype=torch.float16, device='cuda') * scale
        go = torch.ones(N, dtype=torch.float16, device='cuda') * 0.01
        gi = torch.zeros(N, dtype=torch.float16, device='cuda')
        fn(go.data_ptr(), x.data_ptr(), gi.data_ptr(), N)
        torch.cuda.synchronize()
        nan = gi.isnan().any().item()
        inf = gi.isinf().any().item()
        if nan or inf:
            print(f"PROBLEM scale={scale}, N={N}: nan={nan}, inf={inf}")
        else:
            print(f"OK scale={scale}, N={N}")

# 测试 SiLU forward 是否产生 NaN
so_fwd = ctypes.CDLL('./output/full_agent/silu_forward.so')
fn_fwd = so_fwd.launch_kernel
fn_fwd.restype = None
fn_fwd.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]

for scale in [0.01, 1.0, 10.0]:
    x = torch.randn(3072, dtype=torch.float16, device='cuda') * scale
    out = torch.zeros(3072, dtype=torch.float16, device='cuda')
    fn_fwd(x.data_ptr(), out.data_ptr(), 3072)
    torch.cuda.synchronize()
    nan = out.isnan().any().item()
    print(f"SiLU fwd scale={scale}: nan={nan}")
PYEOF
