#!/bin/bash
#SBATCH --job-name=test_patch
#SBATCH --output=logs/test_patch-%j.out
#SBATCH --error=logs/test_patch-%j.err
#SBATCH --partition=fnlp-4090
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

cd /remote-home1/hywei/operator_agent_2
python3 << 'PYEOF'
import sys, torch, torch.nn as nn, ctypes
sys.path.insert(0, '.')
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from examples.full_agent_lora_train import (
    create_custom_silu, create_custom_rmsnorm, patch_model_operators
)

MODEL = '/remote-home1/share/models/Qwen/Qwen3-0.6B'
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map='auto', trust_remote_code=True)

silu_fn, _, _ = create_custom_silu(
    './output/full_agent/silu_forward.so',
    './output/full_agent/silu_backward.so'
)
RMSNormMod, _, _ = create_custom_rmsnorm(
    './output/full_agent/rmsnorm_forward.so',
    './output/full_agent/rmsnorm_backward.so'
)
silu_n, rmsnorm_n = patch_model_operators(model, silu_fn, RMSNormMod)
print(f"Patched: SiLU={silu_n}, RMSNorm={rmsnorm_n}")

# 检查参数总数
total = sum(p.numel() for p in model.parameters())
print(f"Total params: {total:,}")

# 检查前几个 RMSNorm weight 是否在 parameters() 中
rmsnorm_params = 0
for name, p in model.named_parameters():
    if 'weight' in name and 'lm_head' not in name and 'embed' not in name:
        if p.shape[0] in (128, 1024):
            rmsnorm_params += 1
print(f"RMSNorm-like weight params found: {rmsnorm_params}")

# 做一步 forward+backward
device = torch.device('cuda')
tokenizer = __import__('transformers').AutoTokenizer.from_pretrained(MODEL)
inp = tokenizer("hello world", return_tensors='pt')
input_ids = inp['input_ids'].to(device)
out = model(input_ids=input_ids, labels=input_ids)
loss = out.loss
print(f"loss={loss.item():.4f} nan={loss.isnan().item()}")
loss.backward()
# 检查梯度
nan_grads = [(n,p) for n,p in model.named_parameters() if p.grad is not None and p.grad.isnan().any()]
print(f"NaN grads: {len(nan_grads)}")
if nan_grads:
    for n, p in nan_grads[:5]:
        print(f"  NaN in: {n} shape={p.shape}")
else:
    print("All gradients OK!")
PYEOF
