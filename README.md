# Operator Agent — 异构 GPU 算子自动生成系统

> 基于多 Agent 协作的 LLM 驱动系统，为异构 GPU 集群自动生成、编译、验证高性能算子内核。
> 支持 NVIDIA / AMD / 华为昇腾，覆盖从代码生成到分布式训练部署的完整链路。

---

## 系统能力

### 已验证（Phase 1 通过 ✅）

| 能力 | 说明 |
|------|------|
| **CUDA 算子自动生成** | 输入算子名 → LLM 生成 CUDA kernel → nvcc 编译 → 数值验证 |
| **5 种核心算子** | rmsnorm、gelu、silu、flash_attention、matmul 全部通过 |
| **编译错误自修复** | 36 条自动 patch 规则，编译失败时先 auto_fix 再重试 |
| **编译错误知识库** | 自动积累 LLM 常见错误模式，注入 prompt 防止重复 |
| **Few-shot 代码注入** | 从历史成功代码中检索参考实现注入 prompt |
| **代码质量自动降级** | 复杂版本编译失败 ≥2 次后自动切换简单版 prompt |
| **并行算子生成** | 同优先级算子并行生成（rmsnorm/gelu/silu 并行） |
| **LLM 调用重试** | 指数退避重试（2s→4s→8s），支持超时恢复 |
| **SQLite 算子仓库** | 版本追踪、并发安全、历史回溯 |
| **Benchmark 对比** | 自动与 PyTorch 原生实现对比耗时 |

### 框架已实现（待真机验证）

| 能力 | 说明 |
|------|------|
| **HIP 代码生成** | AMD GPU (MI300X/MI250X) 的 HIP kernel 生成 |
| **AscendC 代码生成** | 华为昇腾 (910B/910C) 的 AscendC kernel 生成 |
| **训练代码分析** | 自动提取训练脚本中的算子依赖（支持 LLaMA/GPT/Qwen 等架构） |
| **算子注入训练** | 将生成的 kernel 编译为 .so 并注入训练脚本 |
| **分布式策略** | 异构集群负载均衡、通信后端选择（NCCL/RCCL/UCC） |
| **运行时监控** | GPU 利用率/显存监控（支持 nvidia-smi/rocm-smi/npu-smi） |

---

## 系统架构

```
用户请求（算子名 / 训练代码）
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│              MasterOrchestrator (V2 编排器)                    │
│         双路径：已知 GPU → 查仓库  /  未知 GPU → 完整生成        │
└──┬──────────┬──────────┬───────────┬──────────┬─────────────┘
   │          │          │           │          │
   ▼          ▼          ▼           ▼          ▼
┌──────┐ ┌────────┐ ┌────────┐ ┌─────────┐ ┌──────────┐
│Train │ │  Spec  │ │ Code   │ │ Review  │ │ Distri-  │
│Analyst│ │Analyzer│ │  Gen   │ │  Loop   │ │ bution   │
│(训练 │ │(算子   │ │(CUDA/  │ │(5阶段   │ │(分布式   │
│ 分析)│ │  IR化) │ │HIP/   │ │ 验证    │ │  部署)   │
│      │ │        │ │AscendC)│ │ +修复)  │ │          │
└──────┘ └────────┘ └───┬────┘ └────┬────┘ └──────────┘
                        │           │
                   ┌────▼───────────▼────┐
                   │  编译错误知识库 (KB)   │
                   │  36条自动修复规则      │
                   │  + 动态学习新模式      │
                   └────────┬────────────┘
                            │
                   ┌────────▼────────────┐
                   │  算子仓库 (SQLite)    │
                   │  版本管理 + Few-shot  │
                   └─────────────────────┘
```

### 13 个 Agent

| Agent | 职责 |
|-------|------|
| MasterOrchestrator | 双路径编排（已知GPU查仓库/未知GPU完整生成） |
| TrainingAnalystAgent | 分析训练代码，提取算子依赖 |
| HardwareProfilerAgent | 分析目标 GPU 规格 |
| GPUDiscoveryAgent | 瀑布式查询未知 GPU（本地DB→网络→LLM推断） |
| SDKResolverAgent | 确定每种 GPU 的编程 SDK |
| OperatorSpecAgent | 自然语言 → 标准化 OperatorIR |
| TilingAgent | 根据 GPU 片上内存计算最优分块 |
| CodeGenAgent | 为 CUDA/HIP/SYCL/Triton/AscendC 生成内核代码 |
| OptimizerAgent | 基于 Roofline 模型分析瓶颈并优化 |
| VerifierAgent | 编译检查 + 正确性验证 + 性能验证 |
| ReviewLoopAgent | 5 阶段渐进式验证循环（带上下文历史） |
| DistributionAgent | 异构集群分布式部署方案 |
| TrainingExecutorAgent | 算子注入训练脚本 + 启动命令生成 |
| RuntimeMonitorAgent | GPU 利用率/显存/通信监控 |

### 4 个 MCP Server

| Server | 工具 |
|--------|------|
| GPUSpecMCPServer | GPU 规格查询（本地DB + 网络 + LLM推断） |
| SDKDocsMCPServer | SDK 编程指南 + tiling 模板 |
| OperatorRegistryMCPServer | 算子仓库 CRUD |
| RemoteExecutorMCPServer | 编译/测试/Benchmark（支持本地/Docker/SSH） |

---

## 支持的 GPU

| 厂商 | 型号 | 后端 | 架构 | Phase 1 |
|------|------|------|------|---------|
| NVIDIA | H100 SXM5 | CUDA | Hopper | ✅ |
| NVIDIA | RTX 4090 | CUDA | Ada Lovelace | ✅ 已验证 |
| NVIDIA | A100 SXM4 | CUDA | Ampere | ✅ |
| NVIDIA | RTX 3090 | CUDA | Ampere | ✅ |
| AMD | MI300X | HIP | CDNA3 | 待验证 |
| AMD | MI250X | HIP | CDNA2 | 待验证 |
| Intel | Gaudi 3 | SYCL | Gaudi3 | 待验证 |
| 华为 | 昇腾 910B | AscendC | DaVinci v2 | 待验证 |
| 华为 | 昇腾 910C | AscendC | DaVinci v3 | 待验证 |

跨平台：所有 GPU 均支持 **Triton** 统一后端

---

## 快速开始

### 安装

```bash
git clone <repo_url> && cd operator_agent
pip install -r requirements.txt

# 配置 LLM API
cp .env.example .env
# 编辑 .env，填入 QWEN_API_KEY（或 OPENAI_API_KEY / ANTHROPIC_AUTH_TOKEN）
```

### 三阶段渐进测试

```bash
# Phase 0：本地代码生成质量测试（无需 GPU）
python tests/hetero/hetero_test.py --phase 0

# Phase 1：在 NVIDIA GPU 上编译 + 数值验证
python tests/hetero/hetero_test.py --phase 1 --gpu rtx_4090 --backend cuda --llm qwen

# Phase 2：在 AMD/昇腾 上编译 + 验证（需要对应硬件）
python tests/hetero/hetero_test.py --phase 2 --gpu ascend_910b --backend ascendc --llm qwen
```

### Slurm 集群提交

```bash
# 直接提交（分区和代理已配置好）
sbatch tests/hetero/run_phase1_slurm.sh

# 自定义参数
sbatch --partition=fnlp-4090 --export=GPU_ID=rtx_4090,BACKEND=cuda tests/hetero/run_phase1_slurm.sh
```

### 单算子生成（命令行）

```bash
# 为 H100 + MI300X 混合集群生成 FlashAttention
python main.py --operator "FlashAttention v2" --gpus h100_sxm5 mi300x

# 为 RTX 4090 生成 RMSNorm
python main.py --operator "RMSNorm" --gpus rtx_4090
```

### Python API

```python
import asyncio
from main import run_operator_generation

async def generate():
    results = await run_operator_generation(
        operator_request="FlashAttention v2 with BF16 support",
        target_gpus=["h100_sxm5", "mi300x"],
    )
    if results["success"]:
        for gpu, kernel in results["output"]["kernels"].items():
            print(f"{gpu}: {kernel.backend} kernel, {len(kernel.source_code)} chars")

asyncio.run(generate())
```

---

## 项目结构

```
operator_agent/
├── agents/                        # 14 个专业 Agent
│   ├── base_agent.py              # 基类（含 LLM 重试、状态管理）
│   ├── orchestrator.py            # V1 编排器（串行流水线）
│   ├── code_generator.py          # 代码生成（5 种后端 + 自动降级）
│   ├── review_loop.py             # 5 阶段验证循环（带上下文历史）
│   ├── training_analyst.py        # 训练代码分析
│   ├── training_executor.py       # 算子注入 + 编译 + 启动
│   ├── runtime_monitor.py         # GPU 运行时监控
│   ├── spec_analyzer.py           # 算子规格解析
│   ├── tiling_agent.py            # 分块策略计算
│   ├── optimizer.py               # Roofline 驱动优化
│   ├── verifier.py                # 正确性 + 性能验证
│   ├── distribution.py            # 分布式策略
│   ├── gpu_discovery.py           # 未知 GPU 发现
│   └── sdk_resolver.py            # SDK 选择
│
├── orchestrator_v2.py             # V2 编排器（双路径 + 并行生成）
│
├── models/                        # 核心数据模型
│   ├── hardware_model.py          # GPUSpec / ComputeSpec / MemorySpec
│   └── operator_ir.py             # OperatorIR / GeneratedKernel / TensorSpec
│
├── knowledge_base/                # 知识库
│   ├── hardware_specs/
│   │   ├── gpu_database.py        # 10+ GPU 完整规格数据库
│   │   └── ascend_specs.py        # 昇腾 AI Core 内部规格
│   └── compile_error_kb.py        # 编译错误知识库（36条规则 + 自动学习）
│
├── operators/
│   └── registry.py                # SQLite 算子仓库（版本管理 + Few-shot）
│
├── mcp_servers/                   # 4 个 MCP 工具服务器
│   ├── base_server.py             # MCP 协议基类
│   ├── gpu_spec_server.py         # GPU 规格查询
│   ├── sdk_docs_server.py         # SDK 文档
│   ├── operator_registry_server.py # 算子仓库 MCP 接口
│   └── remote_executor_server.py  # 远程编译/测试/Benchmark
│
├── prompts/
│   └── code_gen_prompts.py        # 4 种后端 Prompt + 降级版 + 版本管理
│
├── tools/
│   ├── llm_client.py              # LLM 客户端（Qwen/OpenAI/Anthropic/Mock）
│   ├── cpu_simulator.py           # CPU 模拟器（20+ 算子参考实现 + Roofline）
│   └── model_router.py            # 多模型路由（简单/复杂算子分流）
│
├── tests/
│   ├── unit/                      # 单元测试
│   ├── simulation/                # 无 GPU 完整模拟测试（7 大部分）
│   └── hetero/
│       ├── hetero_test.py         # 异构 GPU 三阶段测试框架
│       └── run_phase1_slurm.sh    # Slurm 提交脚本
│
├── config/config.yaml             # 全局配置（LLM / 优化 / 验证 / 路由）
├── main.py                        # CLI 入口
└── .env                           # API 密钥配置
```

---

## 核心设计

### 1. 编译错误知识库 — 从失败中学习

```
LLM 生成代码 → 编译失败 → KB 记录错误模式
                              ↓
                 下次生成时注入 prompt："不要使用 __float22half2_rn"
                              ↓
                 同时 auto_fix 自动修复已知错误
```

- 36 条 CUDA 自动修复规则（覆盖 half2 intrinsics、wmma 命名空间、VLA 等）
- 编译失败时自动学习新模式（从 stderr 提取 undefined identifier 等）
- 动态生成禁用列表注入 CodeGen prompt

### 2. 代码质量自动降级

```
第 1 次：生成优化版本（half2 向量化 + shared memory tiling）
  → 编译通过 → 使用 ✅
  → 编译失败 → auto_fix 重试
    → 仍失败 → LLM 重新生成

第 2+ 次失败：自动降级到简单版本
  prompt: "用最简单的方式实现，不用 half2/wmma/shared memory，
          纯 float 精度，每线程处理一个元素"
```

### 3. 双路径编排

```
GPU 列表输入
  ├─ 已知 GPU（仓库里有验证通过的代码）→ 直接复用
  └─ 未知 GPU → 完整生成流程
      ├─ GPU 发现（本地DB → 网络爬虫 → LLM推断）
      ├─ SDK 选择（CUDA/HIP/AscendC/SYCL）
      └─ 并行生成 → ReviewLoop 验证 → 写入仓库
```

### 4. 异构集群负载均衡

基于各 GPU FP16 峰值算力比例分配：
- MI300X（1307 TFLOPs）vs H100（989 TFLOPs）→ 57% : 43%
- 通信后端自动选择：NCCL（NVIDIA）/ RCCL（AMD）/ UCC（混合集群）

---

## 配置

### .env 文件

```bash
# Qwen（阿里云 DashScope）
QWEN_API_KEY=sk-xxx
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3-235b-a22b

# 或 OpenAI
OPENAI_API_KEY=sk-xxx

# 或 Anthropic
ANTHROPIC_AUTH_TOKEN=sk-xxx
ANTHROPIC_BASE_URL=https://api.anthropic.com
```

### config/config.yaml

```yaml
llm:
  backend: "qwen"                    # qwen / openai / anthropic / mock
  temperature: 0.1
  routing:
    enabled: false                   # 多模型路由
    fast_model: "qwen-plus"          # 简单算子
    strong_model: "qwen3-235b-a22b"  # 复杂算子

workflow:
  max_retry_on_failure: 2            # LLM 调用重试次数
  parallel_codegen: true             # 并行代码生成

optimizer:
  max_iterations: 3                  # 最大优化迭代
  target_efficiency: 0.75            # 目标带宽利用率
```

---

## 测试结果

### Phase 1 — RTX 4090 (Slurm, fnlp-4090)

```
  ✅ rmsnorm           compile=✅  math=✅  pytorch=0.13ms  speedup=1.00x
  ✅ gelu              compile=✅  math=✅  pytorch=0.01ms  speedup=1.00x
  ✅ silu              compile=✅  math=✅  pytorch=0.02ms  speedup=1.00x
  ✅ flash_attention   compile=✅  math=✅  pytorch=0.13ms  speedup=1.00x
  ✅ matmul            compile=✅  math=✅  pytorch=0.43ms  speedup=1.00x
```

### Phase 0 — 无 GPU 静态分析

20 个 (算子 × GPU) 组合中 15/20 通过静态质量检查。

---

## 扩展指南

### 添加新 GPU

```python
# knowledge_base/hardware_specs/gpu_database.py
MY_GPU = GPUSpec(
    model_name="My GPU",
    vendor=GPUVendor.NVIDIA,
    compute_capability="9.0",
    # ... 填写规格
)
GPU_DATABASE["my_gpu"] = MY_GPU
```

### 添加新算子模板

```python
# agents/spec_analyzer.py
OPERATOR_TEMPLATES["my_operator"] = {
    "category": OperatorCategory.ELEMENTWISE,
    "math_description": "f(x) = ...",
    "reference_impl": "return F.my_op(x)",
    # ...
}
```

### 添加新编译错误修复规则

```python
# knowledge_base/compile_error_kb.py — 自动学习
# 编译失败时 KB 会自动从 stderr 提取新模式
# 也可以手动添加：
kb.record_error("cuda", 'error: identifier "xxx" is undefined', source_code)
```

---

## 技术栈

- **LLM**: Qwen3 / GPT-4 / Claude（代码生成核心）
- **存储**: SQLite（算子仓库 + 编译错误 KB）
- **框架**: Python asyncio（异步多 Agent 协作）
- **GPU 工具链**: CUDA / ROCm / CANN / Triton
- **通信**: NCCL / RCCL / UCC
- **测试**: pytest + pytest-asyncio
- **调度**: Slurm（支持多分区 + HTTP 代理）
