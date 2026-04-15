# Operator Agent — 异构 GPU 算子自动生成 + 注入训练系统

> 给定一个训练任务，系统自动分析所需算子，调用 LLM 生成 CUDA kernel，编译验证后注入模型，完成端到端 LoRA 微调。

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/) [![CUDA](https://img.shields.io/badge/CUDA-12.x-green)](https://developer.nvidia.com/cuda-toolkit)

---

## 已验证能力

| 能力 | 状态 |
|------|------|
| 训练代码静态分析，自动识别算子依赖 | ✅ |
| LLM 生成 CUDA kernel（forward + backward） | ✅ |
| 编译失败自动 retry（最多 3 次，第 3 次降级） | ✅ |
| ctypes 加载 .so，数值对比验证（相对误差 < 5%） | ✅ |
| 注入 Qwen3-8B（SiLU × 36，RMSNorm × 145） | ✅ |
| Alpaca 指令微调（LoRA），Loss 0.97 → 0.29 | ✅ |
| 按 OperatorCategory 自动判断算子复杂度 | ✅ |
| 新算子自动识别并写入 generated_ops.py | ✅ |
| 验证通过的 kernel 持久化到 SQLite，下次直接复用 | ✅ |

**当前限制：**
- `flash_attention`、`fused_moe` 接口复杂，**暂不支持自动注入**，使用 PyTorch fallback
- 端到端训练流程目前仅支持 CUDA 后端；AscendC/HIP 后端只生成+验证

---

## 快速开始

```bash
git clone https://github.com/hywei0211/operator_agent
cd operator_agent
pip install -r requirements.txt

# 配置 Qwen API（DashScope）
export QWEN_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 端到端训练（最简命令）

```bash
# Custom 模式：全 Agent 生成算子 + 注入 + LoRA 训练（三模式对比）
python examples/full_agent_lora_train.py --llm qwen --mode all --steps 300

# 快速冒烟（Mock LLM，不消耗 API，验证流程正确性）
python examples/full_agent_lora_train.py --llm mock --mode custom --steps 5
```

### CLI 交互模式

```bash
python cli.py generate "帮我生成一个 SiLU 算子，目标 RTX 4090"
python cli.py generate --op rmsnorm --gpu rtx_4090 --backend cuda --llm qwen
python cli.py registry list
python cli.py interactive --llm qwen
```

### Slurm 集群

```bash
MODE=full_agent LLM=qwen STEPS=300 MODEL=/path/to/Qwen3-8B \
    sbatch scripts/slurm/run_lora_slurm.sh
```

---

## 系统流程

```
用户命令
    │
    ▼
Step 0   TrainingAnalystAgent  静态分析，识别 silu/rmsnorm/matmul...
    │
    ▼
Step 0b  AutoOpRegistrar       对比注册表，自动生成缺失算子的 OperatorDesc
    │
    ▼
Step 0c  持久化缓存查询         SQLite 中已有验证通过的 kernel？→ 直接复用
    │
    ▼
Step 1   CodeGenAgent          Qwen API 生成 CUDA kernel（forward + backward）
         编译失败 → stderr 回传 LLM → 重试 ×3（第3次降级为简单版）
    │
    ▼
Step 2   nvcc 编译              .cu → .so，失败时触发 PyTorch fallback
    │
    ▼
Step 2.5 verify_all_kernels    ctypes 加载 → 随机输入 → 对比 PyTorch reference
         通过 → 存入 SQLite；失败 → PyTorch fallback
    │
    ▼
Step 3   patch_model           SiLU × 36 + RMSNorm × 145 注入模型
    │
    ▼
Step 4-5 LoRA 训练 + 评估      Alpaca 指令微调，三模式对比
```

---

## 项目结构

```
operator_agent/
│
├── cli.py                          # CLI 入口（generate/registry/kb/cache/interactive）
├── main.py                         # Python API 入口
├── train.py                        # 训练主入口
├── orchestrator_v2.py              # 双路径编排器
│
├── examples/
│   ├── full_agent_lora_train.py    # 主流程入口（Step 0→5，三模式对比）
│   └── qwen_lora_train.py          # 早期版本（SiLU only，兼容保留）
│
├── operators/                      # 通用算子子系统（核心）
│   ├── op_desc.py                  # OperatorDesc dataclass
│   ├── op_registry.py              # OpRegistry 运行时注册中心 + 全局单例
│   ├── verify.py                   # 通用 verify_kernel（ctypes 数值验证）
│   ├── patch.py                    # 通用 patch_model（4 种注入模式）
│   ├── builtin_ops.py              # 内置算子（silu/rmsnorm/gelu/matmul 等）
│   ├── auto_registrar.py           # 自动识别新算子，生成 OperatorDesc
│   ├── generated_ops.py            # AutoOpRegistrar 写入（运行时生成）
│   └── registry.py                 # SQLite 持久化存储（OperatorEntry）
│
├── agents/
│   ├── training_analyst.py         # 训练代码静态分析
│   ├── spec_analyzer.py            # OperatorIR 生成 + OPERATOR_TEMPLATES
│   ├── code_generator.py           # CUDA/HIP/AscendC kernel 生成 + retry
│   ├── verifier.py                 # 6 级硬件自适应验证
│   ├── review_loop.py              # 5 阶段质量循环
│   └── ...
│
├── knowledge_base/
│   ├── hardware_specs/gpu_database.py   # 10 款 GPU 规格数据库
│   └── compile_error_kb.py              # 37 条编译错误自动 patch 规则
│
├── models/
│   ├── operator_ir.py              # OperatorIR / OperatorCategory
│   └── hardware_model.py           # GPUSpec / GPUBackend
│
├── tools/
│   ├── llm_client.py               # LLM 客户端（Qwen/OpenAI/Mock + 缓存）
│   └── cpu_simulator.py            # CPU reference（20+ 算子）
│
├── prompts/
│   └── code_gen_prompts.py         # Prompt 模板（CUDA/HIP/AscendC/backward）
│
├── scripts/
│   ├── slurm/run_lora_slurm.sh     # Slurm 训练提交脚本
│   ├── tests/test_*.sh             # 算子调试测试脚本（16 个）
│   └── tools/test_qwen.py          # API 连通性测试
│
├── docs/
│   ├── EXAMPLES.md                 # 真实运行示例（含日志）★ 建议先读
│   ├── ARCHITECTURE.md             # 系统架构详解
│   ├── TRAINING.md                 # 端到端训练流程详解
│   ├── OPERATORS.md                # 算子子系统详解
│   ├── WORKFLOW.md                 # 工作流最佳实践
│   └── CLI_REFERENCE.md            # CLI 完整参考手册
│
├── .compile_errors.json            # 编译错误知识库（37 条规则）
└── requirements.txt
```

---

## 支持的算子

| 算子 | Forward | Backward | 模型注入 |
|------|---------|----------|---------|
| silu | ✅ | ✅ float32 输出 | act_fn 替换 |
| rmsnorm | ✅ | ✅ float32 输出 | 模块替换 |
| gelu | ✅ | ✅ float32 输出 | act_fn 替换 |
| matmul | ✅ | — | nn.Linear 替换 |
| softmax | ✅ 验证 | — | — |
| cross_entropy | ✅ 验证 | — | — |
| embedding | ✅ 验证 | — | — |

**AutoOpRegistrar 支持按 OperatorCategory 自动推导接口：**

| OperatorCategory | 自动生成 | ctypes 接口 |
|-----------------|---------|------------|
| ELEMENTWISE | ✅ | `(x*, out*, N)` |
| NORMALIZATION | ✅ | `(x*, w*, out*, N, H, eps)` |
| MATMUL | ✅ | `(A*, B*, C*, M, N, K)` |
| REDUCTION | ✅ | `(x*, out*, N, C)` |
| EMBEDDING | ✅ | `(weight*, idx*, out*, V, H)` |
| ATTENTION / FUSED | ⚠️ 手动实现 | — |

---

## 支持的硬件

| 型号 | Key | 后端 | 端到端训练 |
|------|-----|------|----------|
| RTX 4090 | `rtx_4090` | CUDA | ✅ 默认 |
| H100 SXM5 / PCIe | `h100_sxm5` | CUDA | ✅ |
| A100 80GB | `a100_80gb` | CUDA | ✅ |
| RTX 3090 | `rtx_3090` | CUDA | ✅ |
| MI300X / MI250X | `mi300x` | HIP | 框架已实现 |
| 昇腾 910B / 910C | `ascend_910b` | AscendC | 生成+验证 |

---

## 测试结果（Qwen3-8B，RTX 4090，300 steps）

```
算子覆盖:
  silu_forward    ✅ Agent kernel  rel_err=0.0012
  silu_backward   ✅ Agent kernel  rel_err=0.0000
  rmsnorm_forward ✅ Agent kernel  rel_err=0.0009
  rmsnorm_backward ⚠ PyTorch fallback

Loss: 0.97 → 0.29（下降 70%）

指令跟随评估（5 道测试题）:
  Custom  (Agent 算子):    95.0%  KW: 90%  Fmt: 100%
  Baseline (PyTorch):      92.5%  KW: 85%  Fmt: 100%
  No-finetune (基座):      93.5%  KW: 87%  Fmt: 100%

Custom vs Baseline: +2.5% → 验证 Agent 算子数值等价于 PyTorch
```

---

## 添加新算子

```python
# 方式一：AutoOpRegistrar 自动生成（适合 elementwise/norm/matmul）
from operators.auto_registrar import AutoOpRegistrar
auto_reg = AutoOpRegistrar()
print(auto_reg.explain_complexity("hardswish"))
# → "hardswish: 逐元素算子（简单）— ctypes: (x, out, N)"
desc = auto_reg.generate_op_desc("hardswish")
auto_reg.write_and_register([desc], get_op_registry())

# 方式二：手工定义（适合 forward+backward+注入完整实现）
# 参见 operators/builtin_ops.py 中的 silu/rmsnorm 实现作为模板
```

---

## 详细文档

| 文档 | 内容 |
|------|------|
| [EXAMPLES.md](docs/EXAMPLES.md) | 真实运行示例（含完整日志）**建议先读** |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | 四层架构图、核心数据流 |
| [TRAINING.md](docs/TRAINING.md) | Step 0→5 逐步图解、实测数据 |
| [OPERATORS.md](docs/OPERATORS.md) | OperatorDesc 字段、添加算子步骤 |
| [CLI_REFERENCE.md](docs/CLI_REFERENCE.md) | CLI 完整参考手册 |

---

## License

MIT
