# scripts/

项目脚本目录。

## 子目录

### slurm/
Slurm 集群作业提交脚本。

| 脚本 | 用途 | 典型调用 |
|------|------|---------|
| `run_lora_slurm.sh` | 端到端 LoRA 训练主入口 | `MODE=full_agent LLM=qwen STEPS=300 MODEL=.../Qwen3-8B sbatch scripts/slurm/run_lora_slurm.sh` |

**环境变量：**
```
MODE   = mock / full_agent / full_agent_custom_only / baseline / qwen
LLM    = mock / qwen / openai
STEPS  = 训练步数（默认 344）
MODEL  = 模型路径（默认 Qwen3-0.6B）
```

### tests/
算子验证和梯度调试的 Slurm 测试脚本（开发期使用）。

| 脚本 | 测试内容 |
|------|---------|
| `test_act_grads.sh` | 激活函数梯度测试 |
| `test_silu_bwd.sh` | SiLU backward kernel 验证 |
| `test_rmsnorm_grad.sh` | RMSNorm 梯度对比 |
| `test_rmsnorm_*.sh` | RMSNorm 各种边界条件测试 |
| `test_verify_kernel.sh` | 通用 kernel 数值验证 |
| `test_numerics.sh` | 数值精度测试 |
| `test_gradcheck.sh` | torch.autograd.gradcheck |
| `test_anomaly.sh` | 异常检测（NaN/Inf）|
| `test_forward_nan.sh` | Forward NaN 检测 |
| `test_detect.sh` | 算子注入检测 |
| `test_patch.sh` | 模型算子替换测试 |
| `test_qknorm.sh` | Q/K Norm 测试 |
| `test_qwen_slurm.sh` | Qwen 模型快速验证 |
| `test_real_grads.sh` | 真实训练中的梯度对比 |

### tools/
单次运行工具脚本。

| 文件 | 用途 |
|------|------|
| `test_qwen.py` | 快速测试 Qwen API 连通性 |
