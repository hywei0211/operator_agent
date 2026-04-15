#!/bin/bash
#SBATCH --job-name=anomaly
#SBATCH --output=logs/anomaly-%j.out
#SBATCH --error=logs/anomaly-%j.err
#SBATCH --partition=fnlp-4090
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4

cd /remote-home1/hywei/operator_agent_2
python3 << 'PYEOF'
import sys, torch, torch.nn as nn
sys.path.insert(0, '.')
from transformers import AutoModelForCausalLM, AutoTokenizer
from examples.full_agent_lora_train import create_custom_silu, create_custom_rmsnorm, patch_model_operators
from peft import LoraConfig, get_peft_model

MODEL = '/remote-home1/share/models/Qwen/Qwen3-0.6B'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map='auto', trust_remote_code=True)

# 只用 SiLU custom，不替换 RMSNorm（隔离测试）
silu_fn, sf, sb = create_custom_silu('./output/full_agent/silu_forward.so', './output/full_agent/silu_backward.so')
print(f"SiLU: fwd={sf}, bwd={sb}")

# 只替换 SiLU
from examples.full_agent_lora_train import _SiLUWrapper
for name, module in model.named_modules():
    if hasattr(module, 'act_fn') and 'silu' in type(module.act_fn).__name__.lower():
        module.act_fn = _SiLUWrapper(silu_fn)
        
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"],
                          lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)

device = torch.device('cuda')
text = "The capital of France is Paris"
inp = tokenizer(text, return_tensors='pt', max_length=32, truncation=True)
input_ids = inp['input_ids'].to(device)

with torch.autograd.set_detect_anomaly(True):
    try:
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        nan_grads = sum(1 for n,p in model.named_parameters() if p.grad is not None and p.grad.isnan().any())
        print(f"SiLU-only: loss={out.loss.item():.4f}, nan_grads={nan_grads}")
    except Exception as e:
        print(f"ERROR: {e}")
PYEOF
