#!/bin/bash
#SBATCH --job-name=fwd_nan
#SBATCH --output=logs/fwd_nan-%j.out
#SBATCH --error=logs/fwd_nan-%j.err
#SBATCH --partition=fnlp-4090
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

cd /remote-home1/hywei/operator_agent_2
python3 << 'PYEOF'
import sys, torch, torch.nn as nn, ctypes
sys.path.insert(0, '.')
from transformers import AutoModelForCausalLM, AutoTokenizer
from examples.full_agent_lora_train import (
    create_custom_silu, create_custom_rmsnorm, patch_model_operators
)

MODEL = '/remote-home1/share/models/Qwen/Qwen3-0.6B'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map='auto', trust_remote_code=True)

silu_fn, sf, sb = create_custom_silu('./output/full_agent/silu_forward.so', './output/full_agent/silu_backward.so')
RMSNormMod, rf, rb = create_custom_rmsnorm('./output/full_agent/rmsnorm_forward.so', './output/full_agent/rmsnorm_backward.so')
print(f"Kernels: silu_fwd={sf}, silu_bwd={sb}, rmsnorm_fwd={rf}, rmsnorm_bwd={rb}")

patch_model_operators(model, silu_fn, RMSNormMod)

device = torch.device('cuda')
# 注册 hook 检测 NaN
nan_locations = []
def check_nan_hook(module, input, output):
    if isinstance(output, torch.Tensor) and output.isnan().any():
        nan_locations.append(type(module).__name__)

for m in model.modules():
    m.register_forward_hook(check_nan_hook)

text = "The capital of France is Paris"
inp = tokenizer(text, return_tensors='pt', max_length=32, truncation=True)
input_ids = inp['input_ids'].to(device)
attn_mask = inp['attention_mask'].to(device)

with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
    
print(f"Loss={out.loss.item():.4f}, nan={out.loss.isnan().item()}")
if nan_locations:
    from collections import Counter
    print(f"NaN detected in: {dict(Counter(nan_locations))}")
else:
    print("No NaN in forward pass!")
    
# Now try backward
model.train()
# Remove hooks first
for m in model.modules():
    m._forward_hooks.clear()

from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"], 
                          lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)
out2 = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
out2.loss.backward()
nan_grads = [(n,p) for n,p in model.named_parameters() if p.grad is not None and p.grad.isnan().any()]
print(f"After backward: loss={out2.loss.item():.4f}, nan_grads={len(nan_grads)}")
if nan_grads:
    from collections import Counter
    print(f"First NaN grad: {nan_grads[0][0]}")
PYEOF
