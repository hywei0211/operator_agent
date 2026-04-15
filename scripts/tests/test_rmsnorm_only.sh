#!/bin/bash
#SBATCH --job-name=rmsnorm_only
#SBATCH --output=logs/rmsnorm_only-%j.out
#SBATCH --error=logs/rmsnorm_only-%j.err
#SBATCH --partition=fnlp-4090
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4

cd /remote-home1/hywei/operator_agent_2
python3 << 'PYEOF'
import sys, torch, torch.nn as nn
sys.path.insert(0, '.')
from transformers import AutoModelForCausalLM, AutoTokenizer
from examples.full_agent_lora_train import create_custom_rmsnorm
from peft import LoraConfig, get_peft_model

MODEL = '/remote-home1/share/models/Qwen/Qwen3-0.6B'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map='auto', trust_remote_code=True)

RMSNormMod, rf, rb = create_custom_rmsnorm('./output/full_agent/rmsnorm_forward.so', './output/full_agent/rmsnorm_backward.so')
print(f"RMSNorm: fwd={rf}, bwd={rb}")

# 只替换 RMSNorm
def replace_rmsnorm(parent, prefix=""):
    count = 0
    for child_name, child in list(parent.named_children()):
        ctype = type(child).__name__
        if 'RMSNorm' in ctype:
            new_mod = RMSNormMod(child.weight, child.variance_epsilon)
            setattr(parent, child_name, new_mod)
            count += 1
        else:
            count += replace_rmsnorm(child, f"{prefix}.{child_name}")
    return count

n = replace_rmsnorm(model)
print(f"Replaced {n} RMSNorm modules")

lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"],
                          lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)

device = torch.device('cuda')
text = "The capital of France is Paris"
inp = tokenizer(text, return_tensors='pt', max_length=32, truncation=True)
input_ids = inp['input_ids'].to(device)

out = model(input_ids=input_ids, labels=input_ids)
print(f"Forward loss={out.loss.item():.4f}, nan={out.loss.isnan().item()}")
out.loss.backward()
nan_grads = [(n,p) for n,p in model.named_parameters() if p.grad is not None and p.grad.isnan().any()]
print(f"NaN grads: {len(nan_grads)}")
if nan_grads:
    print(f"First: {nan_grads[0][0]}")
else:
    print("All gradients OK!")
PYEOF
