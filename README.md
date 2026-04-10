# Operator Agent — 异构 GPU 算子自动生成系统

> 一句话解释：你告诉它"帮我写个 SiLU 算子，跑在昇腾 910B 上"，它就能自动生成、编译、验证一个高性能的 GPU 内核代码——包括前向和反向传播。

---

## 这个项目是做什么的？

在深度学习训练中，底层的计算单元叫做 **算子**（Operator）。比如 GELU 激活函数、矩阵乘法、注意力机制（Attention）等。这些算子通常需要针对不同的 GPU 硬件（NVIDIA、AMD、华为昇腾）手写高性能代码，这是一项非常专业且耗时的工作。

本项目通过 **多个 AI Agent 协作** 的方式，自动完成这个过程：

```
你输入: "帮我生成一个 FlashAttention 算子，目标是 H100 和昇腾 910B"
          ↓
系统自动: 理解需求 → 分析硬件 → 生成 forward+backward 代码 → 编译 → 验证正确性 → 优化性能 → 存入仓库
          ↓
你得到: 针对两种 GPU 各自优化的高性能算子代码（可直接用于训练）
```

### 核心亮点

- **多硬件支持**：一套系统同时生成 NVIDIA (CUDA)、AMD (HIP)、华为昇腾 (AscendC) 的代码
- **前向+反向传播**：7 个核心算子同时生成 forward 和 backward kernel，自动生成 `torch.autograd.Function` 包装，可直接用于训练
- **硬件自适应验证**：自动检测当前硬件，有 GPU 就做完整验证，没有就做静态分析+CPU 数学验证（6 级验证体系）
- **全自动流水线**：从自然语言描述到可运行的 GPU 内核，全程无需人工介入
- **自我修复**：编译失败时自动分析错误、修复代码、重试，并记录到知识库避免重复犯错
- **LLM 响应缓存**：相同请求不重复调用 LLM，节省时间和费用
- **断点续传**：ReviewLoop 验证过程支持断点恢复，crash 后不丢失进度
- **算子仓库**：已验证的算子自动入库（含验证等级标注），下次直接复用

---

## 已验证的能力

| 能力 | 说明 |
|------|------|
| **CUDA 算子生成** | 输入算子名 → LLM 生成 CUDA kernel → nvcc 编译 → 数值验证 |
| **AscendC 算子生成** | 输入算子名 → LLM 生成 AscendC kernel → torch_npu 验证 → NPU Benchmark |
| **7 种核心算子** | gelu、silu、rmsnorm、flash_attention、matmul、softmax、fused_moe |
| **前向+反向传播** | 7 个算子全部支持 backward，gradcheck 验证通过 |
| **NPU 真机验证** | 昇腾 910B4 (CANN 8.3.RC1) 上 5/5 算子数值验证通过 |
| **6 级验证体系** | 静态分析 → LLM 审查 → CPU 数学 → 编译 → 硬件运行 → Benchmark |
| **编译错误自修复** | 37 条自动 patch 规则 + 动态学习新模式 |
| **LLM 调用缓存** | SQLite 缓存，7 天过期，避免重复花钱 |
| **CLI 工具** | 自然语言 + 仓库管理/搜索 + 知识库管理 + 交互模式 |

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

### 用自然语言生成算子

```bash
# 自然语言（系统自动解析意图 + ReviewLoop 验证）
python cli.py generate "帮我生成 SiLU 算子，目标昇腾 910B"
python cli.py generate "Generate FlashAttention for H100"

# 指定参数
python cli.py generate --op silu --gpu ascend_910b --backend ascendc --llm qwen

# 快速生成（跳过 ReviewLoop 验证）
python cli.py generate --op silu --gpu rtx_4090 --no-review

# 交互模式（推荐，支持多轮追问）
python cli.py interactive --llm qwen
```

### NPU 测试

```bash
python cli.py npu-test --llm qwen                    # 全部算子
python cli.py npu-test --llm qwen --ops silu gelu     # 指定算子
bash tests/hetero/run_phase1_npu.sh qwen               # Shell 脚本
```

### 算子仓库管理

```bash
python cli.py registry list                              # 列出所有算子
python cli.py registry show silu ascend_910b             # 查看算子详情
python cli.py registry stats                             # 仓库统计
python cli.py registry history silu ascend_910b          # 版本历史
python cli.py registry search --gpu h100_sxm5            # 搜索 H100 上的算子
python cli.py registry search --backend cuda --min-bw 0.6  # 搜索高性能 CUDA 算子
python cli.py registry export ./backup.json              # 导出仓库
```

### 知识库 & 缓存管理

```bash
python cli.py kb stats                                   # 编译错误知识库统计
python cli.py kb export ./kb_share.json                  # 导出（可提交 git 分享）
python cli.py kb import ./kb_from_teammate.json          # 导入
python cli.py cache stats                                # LLM 缓存统计
python cli.py cache clear                                # 清空缓存
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

## 支持的硬件

| 厂商 | 型号 | 编程后端 | 状态 |
|------|------|---------|------|
| NVIDIA | H100 SXM5 | CUDA | ✅ 已验证 |
| NVIDIA | RTX 4090 | CUDA | ✅ 已验证 |
| NVIDIA | A100 80GB | CUDA | ✅ |
| NVIDIA | RTX 3090 | CUDA | ✅ |
| AMD | MI300X | HIP | 框架已实现 |
| AMD | MI250X | HIP | 框架已实现 |
| Intel | Gaudi 3 | SYCL | 框架已实现 |
| 华为 | 昇腾 910B | AscendC | ✅ 已验证 |
| 华为 | 昇腾 910C | AscendC | 框架已实现 |

---

## 验证体系 — 6 级硬件自适应

系统自动检测当前硬件，能验多深就验多深：

| Level | 名称 | 需要什么 | 验证了什么 |
|-------|------|---------|-----------|
| 1 | `static` | 无 | 代码结构、语法、关键 API 使用 |
| 2 | `llm_review` | LLM API | 数学逻辑正确性（LLM 审查） |
| 3 | `cpu_math` | PyTorch | forward + backward 数值对比（gradcheck） |
| 4 | `compiled` | 匹配的编译器 | 真实编译通过 |
| 5 | `hw_verified` | **匹配的 GPU** | 真实硬件运行 + 数值验证 |
| 6 | `benchmarked` | **匹配的 GPU** | + 性能 benchmark |

```
本地有 4090, 生成 CUDA 算子  → 验到 Level 6 (benchmarked)
本地有 4090, 生成昇腾算子    → 验到 Level 3 (cpu_math)
本地什么都没有               → 验到 Level 2 (llm_review)
```

---

## 测试结果

### RTX 4090 (CUDA)

```
  ✅ rmsnorm           compile=✅  math=✅  speedup=1.00x
  ✅ gelu              compile=✅  math=✅  speedup=1.00x
  ✅ silu              compile=✅  math=✅  speedup=1.00x
  ✅ flash_attention   compile=✅  math=✅  speedup=1.00x
  ✅ matmul            compile=✅  math=✅  speedup=1.00x
```

### 昇腾 910B4 (AscendC)

```
  ✅ gelu              err=0.00e+00  time=41.9s  speedup=1.02x  (156 lines)
  ✅ silu              err=0.00e+00  time=46.7s  speedup=1.00x  (193 lines)
  ✅ rmsnorm           err=0.00e+00  time=49.8s  speedup=1.00x  (192 lines)
  ✅ matmul            err=0.00e+00  time=34.6s  speedup=0.99x  (148 lines)
  ✅ flash_attention   err=0.00e+00  time=94.7s  speedup=1.01x  (324 lines)
```

### Backward gradcheck (CPU)

```
  ✅ gelu_backward      gradcheck passed
  ✅ silu_backward      gradcheck passed
  ✅ softmax_backward   gradcheck passed
  ✅ matmul_backward    gradcheck passed
  ✅ rmsnorm_backward   gradcheck passed
```

---

## 项目结构

```
operator_agent/
├── agents/                 # 14 个专业 Agent（核心逻辑）
│   ├── base_agent.py       #   基类（LLM 重试、状态管理、消息通信）
│   ├── orchestrator.py     #   V1 编排器（串行流水线）
│   ├── intent_parser.py    #   意图解析（自然语言→结构化请求）
│   ├── training_analyst.py #   训练代码分析（提取算子依赖）
│   ├── hardware_profiler.py#   硬件分析（GPU 规格查询）
│   ├── gpu_discovery.py    #   未知 GPU 发现（瀑布式查询）
│   ├── sdk_resolver.py     #   SDK 选择（CUDA/HIP/AscendC）
│   ├── spec_analyzer.py    #   算子规格解析（→ OperatorIR）
│   ├── tiling_agent.py     #   分块策略计算
│   ├── code_generator.py   #   代码生成（forward + backward）
│   ├── optimizer.py        #   Roofline 性能优化
│   ├── verifier.py         #   6 级硬件自适应验证
│   ├── review_loop.py      #   5 阶段质量保障循环（含断点续传+进度回调）
│   ├── distribution.py     #   异构集群分布式部署
│   ├── training_executor.py#   算子注入训练（autograd 包装生成）
│   └── runtime_monitor.py  #   GPU 运行时监控
│
├── orchestrator_v2.py      # V2 主调度器（双路径编排 + 并行生成）
├── models/                 # 核心数据模型
│   ├── hardware_model.py   #   GPUSpec / MemorySpec / ComputeSpec
│   └── operator_ir.py      #   OperatorIR（含 backward）/ GeneratedKernel
│
├── knowledge_base/         # 知识库
│   ├── hardware_specs/     #   10+ GPU 完整规格数据库 + 昇腾 AI Core 规格
│   └── compile_error_kb.py #   编译错误知识库（37 条规则 + 自动学习 + 导入导出）
│
├── operators/registry.py   # SQLite 算子仓库（版本管理 + 验证等级 + Few-shot）
├── mcp_servers/            # 4 个 MCP 工具服务器
│   ├── gpu_spec_server.py  #   GPU 规格查询
│   ├── sdk_docs_server.py  #   SDK 文档
│   ├── operator_registry_server.py  # 算子仓库 MCP 接口
│   └── remote_executor_server.py    # 远程编译/测试/Benchmark
│
├── prompts/code_gen_prompts.py  # 代码生成 Prompt（forward + backward + 3 后端）
├── tools/
│   ├── llm_client.py       #   LLM 客户端（Qwen/OpenAI/Claude/Mock + 响应缓存）
│   ├── cpu_simulator.py    #   CPU 模拟器（20+ 算子参考实现 + backward gradcheck）
│   └── model_router.py     #   多模型路由（简单/复杂算子分流）
│
├── tests/                  # 测试
│   ├── unit/               #   单元测试
│   ├── simulation/         #   无 GPU 完整模拟测试
│   └── hetero/             #   异构 GPU 真机测试（CUDA + NPU）
│
├── config/config.yaml      # 全局配置
├── cli.py                  # CLI 入口（generate + registry + kb + cache + interactive）
├── main.py                 # Python API 入口
├── train.py                # V2 训练入口（完整 pipeline）
└── .env                    # API 密钥
```

**详细文档：**

- [系统架构 & 所有组件详解](docs/ARCHITECTURE.md)
- [Agent 完整工作流程](docs/WORKFLOW.md)

---

## 反向传播支持

系统为全部 7 个核心算子生成 backward kernel，支持训练中的梯度计算：

| 算子 | forward | backward | saved_for_backward | gradcheck |
|------|---------|----------|-------------------|-----------|
| gelu | `x*0.5*(1+tanh(...))` | `grad_y * (cdf + x*pdf)` | [x] | ✅ |
| silu | `x*sigmoid(x)` | `grad_y * sig*(1+x*(1-sig))` | [x] | ✅ |
| softmax | `exp(x)/sum(exp(x))` | `y*(grad_y - dot(grad_y,y))` | [y] | ✅ |
| rmsnorm | `x/rms * weight` | chain rule + reduce | [x, weight] | ✅ |
| matmul | `A @ B` | `grad_C@B^T, A^T@grad_C` | [A, B] | ✅ |
| flash_attention | `softmax(QK^T/√d)@V` | recompute P + 3 matmul | [Q, K, V] | ✅ |
| fused_moe | `sum(gate*expert(x))` | routed expert gradients | [h, g, w1, w2] | ✅ |

自动生成 `torch.autograd.Function` 包装，可作为 `F.silu(x)` 的 drop-in 替换：

```python
# 系统自动生成的代码
output = silu_custom(x)        # forward 用自定义 kernel
loss.backward()                # backward 自动调用生成的 backward kernel
```

---

## 配置

### .env 文件

```bash
# Qwen（推荐，阿里云 DashScope）
QWEN_API_KEY=sk-xxx
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3-235b-a22b

# 或 OpenAI
OPENAI_API_KEY=sk-xxx

# 或 Anthropic
ANTHROPIC_AUTH_TOKEN=sk-xxx
```

### config/config.yaml

```yaml
llm:
  backend: "qwen"                    # qwen / openai / anthropic / mock
  model: "qwen3-235b-a22b"
  temperature: 0.1
  enable_thinking: false             # Qwen3 思考模式

optimizer:
  max_iterations: 3
  target_efficiency: 0.75

workflow:
  max_retry_on_failure: 2
  parallel_codegen: true
```

---

## 扩展指南

### 添加新 GPU

```python
# knowledge_base/hardware_specs/gpu_database.py
MY_GPU = GPUSpec(model_name="My GPU", vendor=GPUVendor.NVIDIA, ...)
GPU_DATABASE["my_gpu"] = MY_GPU
```

### 添加新算子模板（含 backward）

```python
# agents/spec_analyzer.py
OPERATOR_TEMPLATES["my_operator"] = {
    "category": OperatorCategory.ELEMENTWISE,
    "math_description": "y = f(x)",
    "reference_impl": "return F.my_op(x)",
    "backward_math_description": "grad_x = grad_y * f'(x)",
    "backward_reference_impl": "return grad_output * f_prime(x)",
    "saved_for_backward": ["x"],
}
```

### 分享编译错误经验

```bash
# 导出本机积累的编译错误经验
python cli.py kb export ./my_errors.json

# 团队成员导入
python cli.py kb import ./my_errors.json
```

---

## 技术栈

| 类别 | 技术 |
|------|------|
| LLM | Qwen3 / GPT-4 / Claude（+ SQLite 响应缓存） |
| 存储 | SQLite（算子仓库 + 编译错误 KB + LLM 缓存） |
| 框架 | Python asyncio（异步多 Agent 协作） |
| GPU 工具链 | CUDA / ROCm / CANN (AscendC) / Triton |
| 梯度验证 | torch.autograd.gradcheck（有限差分法） |
| NPU 验证 | torch_npu + CANN 8.3.RC1 |
| CLI | Click |
| 测试 | pytest + 自定义 NPU 测试框架 |
