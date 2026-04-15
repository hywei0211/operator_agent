#!/bin/bash
#SBATCH --job-name=rmsnorm_scale
#SBATCH --output=logs/rmsnorm_scale-%j.out
#SBATCH --error=logs/rmsnorm_scale-%j.err
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

# 模拟真实训练时的梯度规模
# loss 从 9.5 开始，grad_norm 约 55，这些梯度很大
# 在 fp16 范围内测试
for scale in [1.0, 10.0, 50.0]:
    N, H = 64, 1024
    x = torch.randn(N, H, dtype=torch.float16, device='cuda') * 2.0  # 真实模型激活值
    w = torch.ones(H, dtype=torch.float16, device='cuda')
    go = torch.randn(N, H, dtype=torch.float16, device='cuda') * scale  # 上游梯度
    gx = torch.zeros(N, H, dtype=torch.float16, device='cuda')
    gw = torch.zeros(H, dtype=torch.float32, device='cuda')
    
    fn_bwd(go.data_ptr(), x.data_ptr(), w.data_ptr(), gx.data_ptr(), gw.data_ptr(), N, H, 1e-6)
    torch.cuda.synchronize()
    
    inf = gx.isinf().any().item()
    nan = gx.isnan().any().item()
    print(f"scale={scale}: gx nan={nan}, inf={inf}, max={gx.abs().max().item():.1f}")
    
    # reference
    xr = x.float().requires_grad_(True)
    wr = w.float().requires_grad_(True)
    rms = torch.sqrt(xr.pow(2).mean(-1,keepdim=True)+1e-6)
    y = xr/rms*wr
    y.backward(go.float())
    ref = xr.grad.half()
    print(f"  ref: max={ref.abs().max().item():.1f}, inf={ref.isinf().any().item()}")
PYEOF
