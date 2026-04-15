#!/bin/bash
#SBATCH --job-name=real_grads
#SBATCH --output=logs/real_grads-%j.out
#SBATCH --error=logs/real_grads-%j.err
#SBATCH --partition=fnlp-4090
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4

cd /remote-home1/hywei/operator_agent_2
python3 << 'PYEOF'
import sys, torch, torch.nn as nn, ctypes
sys.path.insert(0, '.')
from transformers import AutoModelForCausalLM, AutoTokenizer
from examples.full_agent_lora_train import create_custom_rmsnorm, RMSNormCustomModule as DummyClass

MODEL = '/remote-home1/share/models/Qwen/Qwen3-0.6B'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map='auto', trust_remote_code=True)

RMSNormMod, rf, rb = create_custom_rmsnorm('./output/full_agent/rmsnorm_forward.so', './output/full_agent/rmsnorm_backward.so')

# 注册 hook 捕获实际 backward 中的 grad
grad_info = {}
class RMSNormDebug(RMSNormMod):
    def __init__(self, weight, eps):
        # RMSNormMod 是一个 class，需要正确调用
        pass

# 手动替换并记录实际梯度
from examples.full_agent_lora_train import create_custom_rmsnorm, _SiLUWrapper
RMSNormMod, rf, rb = create_custom_rmsnorm('./output/full_agent/rmsnorm_forward.so', './output/full_agent/rmsnorm_backward.so')

# 只替换第一层的 input_layernorm 进行测试
target = model.model.layers[0].input_layernorm
from examples.full_agent_lora_train import create_custom_rmsnorm
_, _, _ = create_custom_rmsnorm('./output/full_agent/rmsnorm_forward.so', './output/full_agent/rmsnorm_backward.so')
# 重新创建
RMSNormMod2, _, _ = create_custom_rmsnorm('./output/full_agent/rmsnorm_forward.so', './output/full_agent/rmsnorm_backward.so')
new_rms = RMSNormMod2(target.weight, target.variance_epsilon)
model.model.layers[0].input_layernorm = new_rms

device = torch.device('cuda')
text = "The capital of France"
inp = tokenizer(text, return_tensors='pt', max_length=16, truncation=True)
input_ids = inp['input_ids'].to(device)

# 手动 backward，检查第一层 RMSNorm 的梯度
out = model(input_ids=input_ids, labels=input_ids)
print(f"loss={out.loss.item():.4f}")

# 检查 RMSNorm weight 的梯度
out.loss.backward()
w = model.model.layers[0].input_layernorm.weight
if w.grad is not None:
    print(f"RMSNorm[0] weight.grad: nan={w.grad.isnan().any().item()}, max={w.grad.abs().max().item():.4f}")
else:
    print("RMSNorm[0] weight.grad is None")
    
# 检查前一层 embed 的梯度
emb_w = model.model.embed_tokens.weight
if emb_w.grad is not None:
    print(f"embed_tokens grad: nan={emb_w.grad.isnan().any().item()}")
PYEOF
