#!/bin/bash
#SBATCH --job-name=act_grads
#SBATCH --output=logs/act_grads-%j.out
#SBATCH --error=logs/act_grads-%j.err
#SBATCH --partition=fnlp-4090
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

cd /remote-home1/hywei/operator_agent_2
python3 << 'PYEOF'
import sys, torch, torch.nn as nn, ctypes
sys.path.insert(0, '.')
from transformers import AutoModelForCausalLM, AutoTokenizer
from examples.full_agent_lora_train import create_custom_rmsnorm

MODEL = '/remote-home1/share/models/Qwen/Qwen3-0.6B'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map='auto', trust_remote_code=True)

# 只替换第一个 input_layernorm
from examples.full_agent_lora_train import create_custom_rmsnorm

so_bwd = ctypes.CDLL('./output/full_agent/rmsnorm_backward.so')
fn_bwd = so_bwd.launch_kernel
fn_bwd.restype = None
fn_bwd.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float]

# 先做一次 baseline forward/backward，捕获真实激活值
acts = {}
def capture_hook(name):
    def hook(m, inp, out):
        acts[name] = (inp[0].detach().clone(), out.detach().clone())
    return hook

rn0 = model.model.layers[0].input_layernorm
rn0.register_forward_hook(capture_hook('rn0'))

device = torch.device('cuda')
text = "The capital of France is Paris"
inp = tokenizer(text, return_tensors='pt', max_length=32, truncation=True)
input_ids = inp['input_ids'].to(device)

out = model(input_ids=input_ids, labels=input_ids)
out.loss.backward()

print(f"Baseline: loss={out.loss.item():.4f}")
x_real, y_real = acts['rn0']
print(f"Real x: shape={x_real.shape}, max={x_real.abs().max().item():.4f}, nan={x_real.isnan().any()}")

# 用真实激活值测试 RMSNorm backward
rn0_w = rn0.weight
eps = rn0.variance_epsilon
x2d = x_real.reshape(-1, x_real.shape[-1])
N, H = x2d.shape
print(f"N={N}, H={H}")

# 假设 upstream grad 都是 0.01
go = torch.ones(N, H, dtype=torch.float16, device='cuda') * 0.01
gx = torch.zeros(N, H, dtype=torch.float16, device='cuda')
gw = torch.zeros(H, dtype=torch.float32, device='cuda')
x_cont = x2d.half().contiguous()
w_cont = rn0_w.half().contiguous()
fn_bwd(go.data_ptr(), x_cont.data_ptr(), w_cont.data_ptr(), gx.data_ptr(), gw.data_ptr(), N, H, float(eps))
torch.cuda.synchronize()
print(f"Custom gx: nan={gx.isnan().any()}, inf={gx.isinf().any()}, max={gx.abs().max().item():.6f}")

# Reference
xr = x_cont.float().requires_grad_(True)
wr = w_cont.float().requires_grad_(True)
rms = torch.sqrt(xr.pow(2).mean(-1,keepdim=True)+eps)
y = xr/rms*wr
y.backward(go.float())
ref_gx = xr.grad.half()
print(f"Ref gx: nan={ref_gx.isnan().any()}, inf={ref_gx.isinf().any()}, max={ref_gx.abs().max().item():.6f}")
print(f"Max diff: {(gx-ref_gx).abs().max().item():.6f}")
PYEOF
