#!/bin/bash
#SBATCH --job-name=detect
#SBATCH --output=logs/detect-%j.out
#SBATCH --error=logs/detect-%j.err
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
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map='auto', trust_remote_code=True)

RMSNormMod, rf, rb = create_custom_rmsnorm('./output/full_agent/rmsnorm_forward.so', './output/full_agent/rmsnorm_backward.so')
def replace_rmsnorm(parent):
    count = 0
    for cn, child in list(parent.named_children()):
        if 'RMSNorm' in type(child).__name__:
            setattr(parent, cn, RMSNormMod(child.weight, child.variance_epsilon))
            count += 1
        else:
            count += replace_rmsnorm(child)
    return count
n = replace_rmsnorm(model)
print(f"Replaced {n} RMSNorm")

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
        nan_g = [(n,p) for n,p in model.named_parameters() if p.grad is not None and p.grad.isnan().any()]
        print(f"loss={out.loss.item():.4f}, nan_grads={len(nan_g)}")
        if nan_g:
            # 找第一个 non-lora nan
            for name, p in nan_g[:3]:
                print(f"  NaN: {name}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ANOMALY: {e}")
PYEOF
