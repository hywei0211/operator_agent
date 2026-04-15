#!/bin/bash
#SBATCH --job-name=test_numerics
#SBATCH --output=logs/numerics-%j.out
#SBATCH --error=logs/numerics-%j.err
#SBATCH --partition=fnlp-4090
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

cd /remote-home1/hywei/operator_agent_2
export PATH=/usr/local/cuda/bin:$PATH

python3 -c "
import sys, torch, ctypes
sys.path.insert(0, '.')
print('GPU:', torch.cuda.get_device_name(0))

# RMSNorm forward
so = ctypes.CDLL('./output/full_agent/rmsnorm_forward.so')
fn = so.launch_kernel
fn.restype = None
fn.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int, ctypes.c_int, ctypes.c_float]

N, H = 4, 16
x = torch.randn(N, H, dtype=torch.float16, device='cuda')
w = torch.ones(H, dtype=torch.float16, device='cuda')
out = torch.zeros(N, H, dtype=torch.float16, device='cuda')
fn(x.data_ptr(), w.data_ptr(), out.data_ptr(), N, H, 1e-6)
torch.cuda.synchronize()
xf = x.float()
ref = (xf / torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).half()
diff = (out - ref).abs().max().item()
print(f'RMSNorm fwd max_diff={diff:.6f} nan={out.isnan().any().item()}')
print(f'  out[0]: {out[0,:4].tolist()}')
print(f'  ref[0]: {ref[0,:4].tolist()}')

# SiLU backward
so2 = ctypes.CDLL('./output/full_agent/silu_backward.so')
fn2 = so2.launch_kernel
fn2.restype = None
fn2.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]
x2 = torch.randn(64, dtype=torch.float16, device='cuda')
go2 = torch.ones(64, dtype=torch.float16, device='cuda')
gi2 = torch.zeros(64, dtype=torch.float16, device='cuda')
fn2(go2.data_ptr(), x2.data_ptr(), gi2.data_ptr(), 64)
torch.cuda.synchronize()
x2r = x2.float().requires_grad_(True)
torch.nn.functional.silu(x2r).backward(torch.ones(64, device='cuda'))
ref_gi = x2r.grad.half()
diff2 = (gi2 - ref_gi).abs().max().item()
print(f'SiLU bwd max_diff={diff2:.6f} nan={gi2.isnan().any().item()}')
"
echo ""
python3 -c "
import sys, torch, ctypes
sys.path.insert(0, '.')
print('Testing RMSNorm backward...')

so = ctypes.CDLL('./output/full_agent/rmsnorm_backward.so')
fn = so.launch_kernel
fn.restype = None
fn.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float]

N, H = 4, 16
x = torch.randn(N, H, dtype=torch.float16, device='cuda')
w = torch.ones(H, dtype=torch.float16, device='cuda')
go = torch.ones(N, H, dtype=torch.float16, device='cuda')
gx = torch.zeros(N, H, dtype=torch.float16, device='cuda')
gw = torch.zeros(H, dtype=torch.float32, device='cuda')
fn(go.data_ptr(), x.data_ptr(), w.data_ptr(), gx.data_ptr(), gw.data_ptr(), N, H, 1e-6)
torch.cuda.synchronize()
print(f'gx nan={gx.isnan().any().item()}, gw nan={gw.isnan().any().item()}')
print(f'gx[0]: {gx[0,:4].tolist()}')
print(f'gw[:4]: {gw[:4].tolist()}')

# reference
xr = x.float().requires_grad_(True)
wr = w.float().requires_grad_(True)
rms = torch.sqrt(xr.pow(2).mean(-1, keepdim=True) + 1e-6)
y = xr / rms * wr
y.backward(torch.ones(N, H, device='cuda'))
ref_gx = xr.grad.half()
ref_gw = wr.grad
print(f'ref gx[0]: {ref_gx[0,:4].tolist()}')
print(f'ref gw[:4]: {ref_gw[:4].tolist()}')
diff_gx = (gx - ref_gx).abs().max().item()
diff_gw = (gw - ref_gw.float()).abs().max().item()
print(f'gx max_diff={diff_gx:.6f}, gw max_diff={diff_gw:.6f}')
"
