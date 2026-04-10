# 系统架构 & 所有组件详解

> 本文档面向想要理解本系统内部构造的读者。不需要任何 GPU 编程背景，会尽量用类比来解释。

---

## 目录

- [整体架构一览](#整体架构一览)
- [打个比方：这个系统像什么？](#打个比方这个系统像什么)
- [一、Agent 层 — 14 个"员工"](#一agent-层--14-个员工)
- [二、数据模型层 — 统一语言](#二数据模型层--统一语言)
- [三、知识库层 — 记忆与经验](#三知识库层--记忆与经验)
- [四、MCP 工具服务器层 — 外部能力](#四mcp-工具服务器层--外部能力)
- [五、工具层 — 基础设施](#五工具层--基础设施)
- [六、Prompt 层 — 与 LLM 对话的艺术](#六prompt-层--与-llm-对话的艺术)
- [七、后端层 — 硬件适配](#七后端层--硬件适配)
- [八、入口层 — 用户接口](#八入口层--用户接口)
- [九、配置与存储](#九配置与存储)
- [十、测试体系](#十测试体系)

---

## 整体架构一览

```
┌─────────────────────────────────────────────────────────────────────┐
│                          用户接口层                                  │
│    cli.py (命令行工具)    main.py (Python API)    train.py (训练入口) │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                        Agent 编排层                                  │
│                                                                      │
│   orchestrator_v2.py (V2 主调度器 — 双路径编排)                        │
│   agents/orchestrator.py (V1 主调度器 — 串行流水线)                    │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  14 个专业 Agent（每个负责一项具体工作）                        │   │
│   │  intent_parser → training_analyst → hardware_profiler →     │   │
│   │  gpu_discovery → sdk_resolver → spec_analyzer →             │   │
│   │  tiling_agent → code_generator → optimizer →                │   │
│   │  verifier → review_loop → distribution →                    │   │
│   │  training_executor → runtime_monitor                        │   │
│   └─────────────────────────────────────────────────────────────┘   │
└──────────┬───────────────────┬───────────────────┬─────────────────┘
           │                   │                   │
┌──────────▼───────┐ ┌────────▼─────────┐ ┌──────▼────────────────┐
│   数据模型层      │ │   知识库层        │ │   工具层               │
│   models/        │ │   knowledge_base/ │ │   tools/              │
│   · OperatorIR   │ │   · GPU 数据库    │ │   · LLM 客户端(+缓存) │
│   · GPUSpec      │ │   · 编译错误 KB   │ │   · CPU 模拟器        │
│   · GeneratedKernel│ │                 │ │   · 模型路由器         │
└──────────────────┘ └──────────────────┘ └───────────────────────┘
           │                   │                   │
┌──────────▼───────────────────▼───────────────────▼─────────────────┐
│                       MCP 工具服务器层                                │
│   mcp_servers/                                                       │
│   · GPUSpecServer (GPU规格查询)    · SDKDocsServer (SDK文档)          │
│   · OperatorRegistryServer (算子仓库)  · RemoteExecutorServer (远程编译)│
└──────────────────────────────────────────────────────────────────────┘
           │
┌──────────▼──────────────────┐  ┌─────────────────────────────────┐
│   后端层 backends/           │  │   存储层                         │
│   · AscendC 代码生成器       │  │   · .operator_registry.db (SQLite)│
│   · CUDA / HIP / Triton     │  │   · .compile_errors.json         │
└─────────────────────────────┘  │   · .llm_cache.db (SQLite)       │
                                 └─────────────────────────────────┘
```

---

## 打个比方：这个系统像什么？

想象一个**软件外包公司**，客户说"我要一个能跑在华为芯片上的高性能计算模块"：

| 角色 | 对应的 Agent | 做什么 |
|------|-------------|--------|
| **项目经理** | MasterOrchestrator | 接单、拆任务、分配工作、把控全局进度 |
| **需求分析师** | IntentParser | 听客户说话，理解他到底想要什么 |
| **训练工程师** | TrainingAnalyst | 分析客户的训练代码，列出需要哪些算子 |
| **硬件工程师** | HardwareProfiler + GPUDiscovery | 查明目标硬件的详细规格 |
| **SDK 顾问** | SDKResolver | 确定用哪种编程语言/框架来写代码 |
| **架构师** | SpecAnalyzer | 把需求变成精确的技术规格书 |
| **性能专家** | TilingAgent | 计算数据怎么切块才能最快处理 |
| **码农** | CodeGenerator | 实际写代码（前向 + 反向内核） |
| **性能调优师** | Optimizer | 真实硬件 profiling + Roofline 分析，优化代码 |
| **测试工程师** | Verifier | 6 级硬件自适应验证（从静态分析到真实 Benchmark） |
| **QA 主管** | ReviewLoop | 统筹 5 轮质量检查，支持断点续传、进度回调 |
| **运维工程师** | Distribution | 规划怎么部署到多台机器上 |
| **部署工程师** | TrainingExecutor | 把算子塞进训练脚本、自动生成 autograd 包装 |
| **监控员** | RuntimeMonitor | 盯着训练过程，发现异常就报警（查不到数据返回 -1） |

---

## 一、Agent 层 — 14 个"员工"

所有 Agent 都继承自 `agents/base_agent.py` 中的 `BaseAgent` 基类。

**模块导出**：`agents/__init__.py` 导出全部 14 个 Agent + `AgentContext`、`AgentResult`、`AgentStatus`、`VerificationLevel`、`HardwareDetector` 等关键类，方便外部 `from agents import *` 一站式引入。

### 1.1 BaseAgent — 所有 Agent 的"基因"

**文件**: `agents/base_agent.py`

每个 Agent 都具备以下能力：

```
BaseAgent
├── run()           → 主执行逻辑（每个子类自己实现）
├── call_llm()      → 调用大语言模型（带自动重试：2s→4s→8s 指数退避）
├── set_status()    → 状态管理（idle → running → completed/failed）
├── send_message()  → Agent 间通信
├── get_system_prompt() → 定义"你是谁"（角色 prompt）
└── success_result() / failure_result() → 统一的结果封装
```

关键数据结构：

| 类 | 用途 | 类比 |
|----|------|------|
| `AgentContext` | 所有 Agent 共享的"黑板"，存放中间产物 | 项目共享文档 |
| `AgentResult` | Agent 执行结果的统一格式 | 工作汇报 |
| `AgentMessage` | Agent 间通信的消息 | 内部邮件 |

---

### 1.2 IntentParser — 意图解析器

**文件**: `agents/intent_parser.py`  
**角色**: 需求分析师  
**一句话**: 把你说的自然语言变成结构化的任务描述

**工作方式**：
```
用户: "帮我写个 SiLU 算子，跑在昇腾 910B 上"
  ↓ LLM 解析
IntentParser 输出:
  {
    "status": "ready",
    "operator": "silu",
    "gpus": ["ascend_910b"],
    "backend": "ascendc"
  }
```

**智能追问**：如果你说的信息不完整，它会追问：
```
用户: "写个 RoPE"
IntentParser: "🤔 请问目标硬件是什么？支持: 昇腾 910B、RTX 4090、H100..."
```

**双保险**：LLM 解析失败时，自动降级到关键词匹配（规则引擎兜底）。

---

### 1.3 TrainingAnalystAgent — 训练代码分析器

**文件**: `agents/training_analyst.py`  
**角色**: 训练工程师  
**一句话**: 分析你的训练代码，告诉系统需要生成哪些算子

**工作方式**：
1. **静态分析**：用正则表达式扫描代码，识别 `F.gelu`、`RMSNorm`、`flash_attention` 等关键词
2. **架构识别**：识别出你用的是 LLaMA / GPT / Qwen 等哪种模型架构，自动补全该架构需要的算子
3. **超参提取**：提取 batch_size、seq_length、hidden_size 等配置
4. **LLM 增强**：用大模型补充静态分析可能遗漏的信息

**输出** `TrainingPlan`：
```
critical_operators: [flash_attention, matmul, rmsnorm]     ← 必须有，优先生成
required_operators: [silu, embedding, softmax]              ← 需要有
optional_operators: [dropout]                               ← 可有可无
model_architecture: "llama"
uses_distributed: True
```

---

### 1.4 HardwareProfilerAgent — 硬件分析器

**文件**: `agents/hardware_profiler.py`  
**角色**: 硬件工程师  
**一句话**: 查明每种 GPU 的详细规格（算力、显存带宽、互联拓扑等）

**工作方式**：
1. 从内置 GPU 数据库查找（10+ 种 GPU 的完整规格）
2. 如果数据库里没有 → 用 LLM 识别最接近的已知 GPU
3. 分析集群特性：是否异构、各 GPU 性能差异比、是否都支持 Triton

**为什么重要**：代码生成时需要知道目标 GPU 有多少 Shared Memory、什么计算能力（sm_89 / sm_90）等，才能生成针对性优化的代码。

---

### 1.5 GPUDiscoveryAgent — GPU 发现器

**文件**: `agents/gpu_discovery.py`  
**角色**: 硬件侦察兵  
**一句话**: 对于未知的 GPU 型号，通过多种渠道自动发现其规格

**瀑布式查询策略**：
```
步骤 1: 本地数据库查找 → 找到则返回（最快、最可靠）
  ↓ 没找到
步骤 2: 通过 MCP Server 网络查询 → 找到则返回
  ↓ 没找到
步骤 3: 用 LLM 根据型号名推断规格 → 标注低可信度
```

每个发现结果都标注**可信度**（0~1），可信度低的会在后续流程中被标注警告。

---

### 1.6 SDKResolverAgent — SDK 解析器

**文件**: `agents/sdk_resolver.py`  
**角色**: SDK 顾问  
**一句话**: 根据 GPU 厂商确定用什么编程框架，并准备好代码生成所需的上下文

**映射关系**：
```
NVIDIA GPU  → CUDA (nvcc 编译器, __global__ 声明, threadIdx 线程ID)
AMD GPU     → HIP  (hipcc 编译器, wavefront=64, MFMA 矩阵指令)
华为昇腾    → AscendC (手动 DataCopy, Cube 16×16 对齐, 双缓冲必须)
Intel GPU   → SYCL (nd_range 并行, local_accessor 局部内存)
跨平台      → Triton (Python 语法, 自动调参)
```

**输出 `SDKContext`**：包含编程语言、编译器、kernel 声明语法、线程 ID 表达式、共享内存语法、同步原语、分块模板代码等 —— 这些信息会被注入到代码生成的 prompt 中。

---

### 1.7 OperatorSpecAgent — 算子规格解析器

**文件**: `agents/spec_analyzer.py`  
**角色**: 架构师  
**一句话**: 把"GELU 激活函数"这样的描述，变成精确的技术规格书（OperatorIR）

**两种工作模式**：

1. **模板匹配**（快速）：内置了 7 种常见算子的完整模板（`OPERATOR_TEMPLATES`），覆盖：
   - `flash_attention`、`rmsnorm`、`gelu`、`silu`、`matmul`、`softmax`、`fused_moe`
   - 每个模板包括：数学公式、输入输出形状、FLOPs 计算公式、PyTorch 参考实现
   - **新增**：每个模板都包含 `backward_math_description`（反向传播数学公式）、`backward_reference_impl`（反向传播 PyTorch 参考实现）、`saved_for_backward`（前向中需保存的张量列表），为反向内核生成提供完整信息
   - 模板已去重，每个算子只有一份权威定义
   
2. **LLM 解析**（灵活）：对于未知的自定义算子，用大模型从描述中提取规格

**输出 `OperatorIR`**（算子中间表示）：这是整个系统的"通用语言"，后续所有组件都基于它工作。

---

### 1.8 TilingAgent — 分块策略计算器

**文件**: `agents/tiling_agent.py`  
**角色**: 性能专家  
**一句话**: 计算数据该怎么切块，才能最高效地利用 GPU 的片上内存

**为什么需要 Tiling？**

GPU 有一小块超快的"片上缓存"（NVIDIA 叫 Shared Memory，昇腾叫 UB），容量很小（几十 ~ 几百 KB）。大数据必须切成小块，一块一块搬进缓存处理。切块策略直接决定性能好坏。

**不同硬件的策略**：

| 硬件 | 片上缓存 | 关键约束 | 典型配置 |
|------|---------|---------|---------|
| NVIDIA | Shared Memory 48~164KB | warp_size=32 | BLOCK_M=128, BLOCK_N=128, BLOCK_K=32 |
| AMD | LDS 64KB | wavefront=64 | BLOCK_M=128, BLOCK_N=128, BLOCK_K=16 |
| 昇腾 | UB 256KB + L0A/L0B 64KB | Cube 16×16 对齐, 必须双缓冲 | tile_m=256, tile_k=64 |

**输出 `TilingConfig`**：包含推荐配置 + 多组候选配置（从激进到保守），供后续 autotuner 选择最优。

---

### 1.9 CodeGenAgent — 代码生成器

**文件**: `agents/code_generator.py`  
**角色**: 核心码农  
**一句话**: 根据算子规格和目标硬件，用 LLM 生成实际的 GPU 内核代码（前向 + 反向）

**支持 5 种后端**：CUDA、HIP、SYCL、Triton、AscendC

**后端选择策略**：
```
NVIDIA GPU → 优先 CUDA，备选 Triton
AMD GPU    → 优先 HIP，备选 Triton
华为昇腾   → 只能 AscendC
Intel GPU  → 优先 SYCL，备选 Triton
```

**关键机制**：

1. **精心构造的 Prompt**：注入算子数学定义、GPU 规格、编译约束、历史错误教训
2. **Few-shot 注入**：从算子仓库检索历史成功代码作为参考
3. **编译错误知识库**：动态生成"不要用 xxx"的禁用列表，防止重复犯错
4. **自动降级**：如果复杂版本连续编译失败 ≥2 次，自动切换到"简单模式"prompt
5. **多层代码提取**：LLM 返回格式不稳定，系统有 5+ 种策略从响应中提取代码
6. **无 LLM 兜底**：即使没有 LLM，也能生成基础模板代码

**新增 — `generate_backward()` 方法**：

```
generate_backward(context, operator_ir=..., gpu_spec=..., forward_kernel=...)
  ↓
1. 要求 operator_ir 必须含有 backward_math_description
2. 根据后端选择对应 prompt:
   - CUDA → build_cuda_backward_prompt()
   - AscendC → build_ascendc_backward_prompt()
3. LLM 生成 backward kernel 代码
4. 无 LLM 时生成占位模板（含数学公式注释）
5. 返回 GeneratedKernel（backward_source_code）
```

这意味着系统现在可以同时生成前向和反向内核，完整支持自定义算子的训练流程。

---

### 1.10 OptimizerAgent — 性能优化器

**文件**: `agents/optimizer.py`  
**角色**: 性能调优师  
**一句话**: 对生成的内核进行真实硬件 profiling + Roofline 分析，针对性优化

**核心工具 — Roofline 模型**：

```
                    峰值算力
                   ┌────────────────────
                  ╱
性能 (TFLOPs)    ╱  ← 计算受限区域
                ╱
               ╱
──────────────╱────────── ← 内存带宽受限区域
             ╱
            ╱
           └──────────────────────────
              算术强度 (FLOPs/Byte)
```

- **内存带宽受限** → 用向量化加载(float4)、共享内存缓存、预取
- **计算受限** → 用 Tensor Core、循环展开、快速数学函数
- **延迟受限** → 提高 occupancy、Warp 原语、持久化 kernel

**优化策略库**：包含 memory_bound / compute_bound / latency_bound / general 四大类共 16 种优化技术。

**新增 — 真实硬件 profiling**：

```
_profile_kernel(kernel, gpu_spec, operator_ir)
  ↓
1. 调用 HardwareDetector.detect() 检测当前硬件环境
2. 判断是否有匹配的真实 GPU（CUDA/HIP/NPU）
  ├── 有 → _try_real_profiling()
  │     · 获取算子参考实现（从 CPUSimulator.REFERENCE_IMPLS）
  │     · 在真实 GPU 上用 PyTorch 执行，测量实际延时
  │     · 计算真实带宽利用率和 TFLOPs
  │     · 返回真实 profiling 数据
  └── 没有 → 退化为代码特征估算
        · _estimate_bw_utilization()：分析代码中内存访问模式
        · _estimate_compute_utilization()：分析代码中计算密度
        · Roofline 模型预测瓶颈类型
```

**与旧版的关键区别**：旧版使用硬编码的假值（如固定 0.65 带宽利用率），新版先尝试真实 GPU profiling，只在无硬件时才退化为基于代码特征的 Roofline 估算——不再有凭空编造的数值。

---

### 1.11 VerifierAgent — 验证器（6 级硬件自适应验证）

**文件**: `agents/verifier.py`  
**角色**: 测试工程师  
**一句话**: 自动检测硬件环境，分层执行 6 级验证 —— 有什么硬件就做什么级别的测试

**旧版只有三重模拟验证，新版是真正的硬件自适应验证系统。**

**核心组件**：

#### HardwareDetector — 硬件探测器

一次性检测当前机器的所有 GPU 硬件和 SDK 环境，结果全局缓存：

```python
HardwareDetector.detect() → {
    "nvidia_gpu": True/False,    # NVIDIA GPU（通过 torch.cuda.is_available）
    "nvidia_gpu_name": "RTX 4090",
    "nvcc": True/False,          # nvcc 编译器可用
    "amd_gpu": True/False,       # AMD GPU（通过 rocm-smi）
    "hipcc": True/False,         # hipcc 编译器可用
    "npu": True/False,           # 华为昇腾 NPU（通过 torch_npu）
    "npu_name": "Ascend 910B",
    "cann": True/False,          # CANN 编译链可用
    "torch": True/False,         # PyTorch 是否安装
}
```

#### VerificationLevel — 6 级验证等级

```
Level 1: STATIC        ← 静态分析（纯规则检查，不需要任何硬件）
  ↓ 有 LLM
Level 2: LLM_REVIEW    ← LLM 代码审查（让大模型检查代码质量）
  ↓ 有 PyTorch
Level 3: CPU_MATH      ← CPU 数学验证（与 PyTorch 参考实现对比）
  ↓ 有匹配编译器（nvcc/hipcc/cann）
Level 4: COMPILED      ← 真实编译通过（但没在目标硬件上运行）
  ↓ 有匹配 GPU 硬件
Level 5: HW_VERIFIED   ← 真实硬件运行 + 数值正确性验证
  ↓ 硬件运行通过
Level 6: BENCHMARKED   ← 真实硬件 Benchmark 完成（测量延时、带宽利用率）
```

**关键设计**：**每一级都包含前面所有级的检查**。系统根据当前硬件环境自动决定能做到哪一级 —— 无 GPU 的笔记本电脑也能做到 Level 3（CPU 数学验证），有 GPU 的服务器可以做到 Level 6。

**输出 `VerificationReport`**：详细报告包含：
- `verification_level`：本次最高达到的验证等级
- `hardware_detected`：检测到的硬件环境
- 各级别的详细结果（静态分析、LLM 审查、CPU 数学误差、编译结果、硬件运行正确性、Benchmark 数据）
- 修复建议（失败时反馈给 CodeGen 修复）

---

### 1.12 ReviewLoopAgent — 质量保障循环

**文件**: `agents/review_loop.py`  
**角色**: QA 主管（**最核心的质量控制组件**）  
**一句话**: 统筹 5 个阶段的质量检查，不合格就打回重做，最多 5 轮

**五阶段渐进式验证**（成本从低到高）：

```
Stage 1: 静态代码审查     ← 最便宜（LLM 检查 + 规则检查）
  ↓ 通过
Stage 2: 编译检查          ← 较便宜（调用编译器）
  ↓ 通过
Stage 3: 数值正确性验证    ← 中等（需要运行代码）
  ↓ 通过
Stage 4: 性能基准测试      ← 较贵（需要 GPU benchmark）
  ↓ 通过
Stage 5: 综合评审          ← 最终裁定
```

**关键设计**：
- **哪里有问题修哪里**：代码错误 → 反馈给 CodeGenAgent 重写；性能不达标 → 反馈给 OptimizerAgent 优化
- **上下文累积**：每次失败的历史都会传给下一轮，避免重复犯同样的错误
- **编译错误自动学习**：编译失败时自动记录错误模式到知识库
- **超时人工介入**：超过 5 轮仍不合格 → 标记为需要人工审查
- **无论是否通过都存仓库**：当前最优版本始终保存，不浪费之前的工作
- **Stage 3/4 使用 VerificationLevel 自适应阈值**：根据 VerifierAgent 能达到的验证等级，动态调整正确性和性能的通过标准（例如：只有 CPU 验证时放宽阈值，有真实硬件 Benchmark 时使用严格阈值）

**新增 — 断点续传（Checkpoint/Resume）**：

```
每个 Stage 完成后 → 自动保存到 .review_checkpoints/
  ├── 文件名: {op_name}_{gpu_model}.ckpt.json
  ├── 内容: iteration、current_code、iteration_history、timestamp
  └── 1 小时后自动过期

下次启动 → 自动检测是否有可恢复的 checkpoint
  ├── 有（且未过期）→ 从上次断点的 iteration 和代码继续
  └── 无 → 全新开始

全部通过 → 自动清除 checkpoint
```

**用途**：系统 crash 或用户中断后，重新启动时自动从上次进度继续，不浪费已完成的验证工作。

**新增 — 进度回调（Progress Callback）**：

```python
review_loop.set_progress_callback(
    lambda stage, iter, max_iter, passed: print(f"{stage} iter={iter} passed={passed}")
)
```

CLI 注入此回调后，用户可实时看到每个 Stage 的进度和结果。每个 Stage 开始和结束时都会触发 `_emit_progress()` 回调。

---

### 1.13 DistributionAgent — 分布式协调器

**文件**: `agents/distribution.py`  
**角色**: 运维工程师  
**一句话**: 为异构 GPU 集群制定分布式部署方案

**处理的核心问题**：

1. **负载均衡**：不同 GPU 性能差异大，按算力比例分配工作
   - 例：MI300X (1307 TFLOPs) vs H100 (989 TFLOPs) → 57% : 43%
2. **通信后端选择**：
   - 全 NVIDIA → NCCL
   - 全 AMD → RCCL
   - 混合集群 → UCC / MPI
3. **张量分片**：确定每个设备处理数据的哪个部分
4. **通信模式**：AllReduce / AllGather / ReduceScatter

**输出 `DistributionPlan`**：包含每个设备的部署方案、通信配置、负载权重、性能预测。

---

### 1.14 TrainingExecutorAgent — 训练执行器

**文件**: `agents/training_executor.py`  
**角色**: 部署工程师  
**一句话**: 把生成的算子注入训练脚本，生成启动命令

**工作流程**：
1. 从算子仓库加载已验证的算子代码
2. 保存内核源文件到磁盘，尝试编译为 `.so`（CUDA 用 nvcc、HIP 用 hipcc）
3. 生成 PyTorch 自定义算子注册代码（`torch.library`）
4. **新增**：调用 `_generate_autograd_wrappers()` 为每个算子生成 `torch.autograd.Function` 包装
5. 修改训练脚本，替换原有的算子调用
6. 生成分布式启动命令（`torchrun` / `deepspeed`）
7. 启动训练进程（默认 `dry_run=True`，只生成不执行）

**新增 — `_generate_autograd_wrappers()`**：

```python
class SiluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 尝试用自定义 kernel，失败时 fallback 到 PyTorch
        try:
            output = torch.ops.custom.silu_forward(x)
        except Exception:
            output = F.silu(x)          # ← 安全降级
        ctx.save_for_backward(x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 如果有自定义 backward kernel，使用它
        # 否则自动降级到 PyTorch autograd
        ...
```

- 如果 `GeneratedKernel.backward_source_code` 存在 → 生成完整的 forward+backward Function
- 否则 → 只生成 forward 包装，backward 由 PyTorch autograd 自动处理
- 每个 forward 都有 try/except 安全降级，自定义 kernel 出错时 fallback 到 PyTorch 原生实现

**干运行日志**：`dry_run=True` 时输出完整的脚本路径和启动命令，方便用户手动检查后再执行。

---

### 1.15 RuntimeMonitorAgent — 运行时监控

**文件**: `agents/runtime_monitor.py`  
**角色**: 监控员  
**一句话**: 盯着训练过程，发现异常就报警并给出建议

**监控指标**：
- GPU 利用率（持续低于 70% → 报警）
- 显存使用（接近上限 → 报警）
- 通信耗时占比（超过 30% → 报警）
- 负载不均衡（最慢 GPU 拖累整体）
- Loss 曲线（发散或停止下降 → 报警）

**改进 — 诚实的"未知"值**：

旧版在无法查询 GPU 利用率时会静默返回 `0.75`（一个看似正常的假值），可能导致监控误判。新版改为返回 `-1.0` 表示"未知/无法查询"，让下游逻辑能明确区分"正常"和"不可用"。

```python
def _query_gpu_utilization(self, gpu_model: str) -> float:
    # 尝试 nvidia-smi / rocm-smi / npu-smi
    ...
    return -1.0  # 无法查询时返回 -1 表示未知
```

---

## 二、数据模型层 — 统一语言

**目录**: `models/`

这一层定义了整个系统的"通用语言"，所有 Agent 之间通过这些数据结构交流。

### 2.1 OperatorIR — 算子中间表示

**文件**: `models/operator_ir.py`

```python
@dataclass
class OperatorIR:
    name: str                           # 算子名，如 "flash_attention"
    category: OperatorCategory          # 类别：elementwise/matmul/attention/...
    description: str                    # 文字描述
    inputs: list[TensorSpec]            # 输入张量列表（名称+形状+类型）
    outputs: list[TensorSpec]           # 输出张量列表
    math_description: str               # 数学公式，如 "y = x * sigmoid(x)"
    reference_impl: str                 # PyTorch 参考实现
    hyperparams: dict                   # 超参数

    # ── 新增：反向传播 ──
    backward_math_description: str      # 反向传播数学公式
                                        # 如 "grad_x = grad_y * sigmoid(x) * (1 + x*(1-sigmoid(x)))"
    backward_reference_impl: str        # PyTorch backward 参考实现
    saved_for_backward: list[str]       # forward 中需保存的张量名，如 ["x"] 或 ["Q","K","V"]

    # ── 复杂度分析 ──
    flops_formula: str                  # FLOPs计算公式
    memory_reads_formula: str           # 内存读取量公式
    memory_writes_formula: str          # 内存写入量公式
    parallel_strategy: ParallelStrategy # 并行策略
    tags: list[str]                     # 标签
```

这是最核心的数据结构 —— 它完整描述了一个算子"是什么"（数学定义）、"怎么算"（计算特征）和"怎么反向"（梯度计算），但**不涉及任何硬件细节**。所有后端的代码生成都从同一个 OperatorIR 出发。

**新增的三个反向传播字段**使得系统能够自动生成 backward kernel，实现完整的训练支持。

### 2.2 GPUSpec — GPU 硬件规格

**文件**: `models/hardware_model.py`

```python
@dataclass
class GPUSpec:
    model_name: str         # 型号名
    vendor: GPUVendor       # 厂商: nvidia/amd/intel/huawei
    architecture: str       # 架构: Hopper/CDNA3/DaVinci
    compute_units: int      # 计算单元数 (SM/CU/AI Core)
    memory: MemorySpec      # 显存: 容量、带宽、类型
    compute: ComputeSpec    # 算力: FP16/FP32/INT8 TFLOPs
    interconnect: InterconnectSpec  # 互联: NVLink/PCIe/Infinity Fabric
    supported_backends: list[GPUBackend]  # 支持的编程后端
```

### 2.3 GeneratedKernel — 生成的内核

**文件**: `models/operator_ir.py`

```python
@dataclass
class GeneratedKernel:
    operator_name: str
    backend: str                    # cuda / hip / sycl / triton / ascendc
    target_gpu: str
    source_code: str                # 前向 kernel 源代码
    header_code: str                # 头文件（如有）
    build_flags: list[str]          # 编译标志
    launch_config: dict             # 启动配置(grid, block)

    # 性能
    estimated_flops: float
    estimated_bandwidth_utilization: float
    optimizations_applied: list[str]
    iteration: int

    # 验证
    correctness_verified: bool
    benchmark_results: dict
    verification_level: str         # ← 新增：none/static/llm_review/cpu_math/compiled/hw_verified/benchmarked

    # ── 新增：反向传播 ──
    backward_source_code: str       # backward kernel 代码
    backward_build_flags: list[str] # backward kernel 编译标志
```

新增的 `verification_level` 字段记录该内核通过的最高验证等级，`backward_source_code` 和 `backward_build_flags` 用于存储生成的反向内核代码。

---

## 三、知识库层 — 记忆与经验

**目录**: `knowledge_base/`

### 3.1 GPU 数据库

**文件**: `knowledge_base/hardware_specs/gpu_database.py`

预置了 10+ 种 GPU 的完整规格数据，包括：
- NVIDIA: H100 SXM5, H100 PCIe, RTX 4090, A100 80GB, RTX 3090
- AMD: MI300X, MI250X
- Intel: Gaudi 3
- 华为: 昇腾 910B, 910C

每条数据包含数百个参数（算力、带宽、缓存大小、互联规格等）。

### 3.2 昇腾 AI Core 规格

**文件**: `knowledge_base/hardware_specs/ascend_specs.py`

昇腾芯片的内部架构比较特殊（AI Core 有多级缓冲区：UB/L0A/L0B/L0C），需要单独建模。

### 3.3 编译错误知识库

**文件**: `knowledge_base/compile_error_kb.py`

这是系统的"错误记忆"，核心机制：

```
编译失败 → 自动记录错误模式 → 下次生成代码时注入 prompt 避免
```

**功能**：

| 功能 | 说明 |
|------|------|
| `auto_fix(source, backend)` | 自动修复已知错误（36 条 CUDA 规则，覆盖 half2 类型、wmma 命名空间等） |
| `generate_prompt_fragment(backend)` | 生成注入 CodeGen prompt 的"不要做 xxx"列表 |
| `record_error(backend, stderr, source)` | 从编译 stderr 自动提取新的错误模式并存储 |
| `export_patterns(output_path)` | **新增**：导出所有错误模式到 JSON 文件（可提交 git 或分享给团队） |
| `import_patterns(input_path, overwrite)` | **新增**：从 JSON 文件导入错误模式（合并到现有知识库） |
| `stats()` | **新增**：返回知识库统计信息（按后端分类的规则数、触发次数等） |

**存储**：`.compile_errors.json`，JSON 格式持久化。

---

## 四、MCP 工具服务器层 — 外部能力

**目录**: `mcp_servers/`

MCP（Model Context Protocol）是一种工具调用协议。Agent 通过 `MCPClient` 调用各 Server 暴露的工具。

### 4.1 BaseMCPServer & MCPClient — 协议基类

**文件**: `mcp_servers/base_server.py`

定义了统一的工具注册、调用、响应格式。每个 Server 注册若干 `MCPTool`，Agent 通过 `MCPClient.call(server_name, tool_name, **params)` 调用。

### 4.2 GPUSpecMCPServer — GPU 规格查询

**文件**: `mcp_servers/gpu_spec_server.py`

提供工具：
- `search_gpu_spec(model_name)` — 查询 GPU 规格（瀑布式：本地DB → 网络 → LLM）
- `list_all_gpus()` — 列出所有已知 GPU
- `compare_gpus(gpu_a, gpu_b)` — 对比两种 GPU

### 4.3 SDKDocsMCPServer — SDK 文档服务

**文件**: `mcp_servers/sdk_docs_server.py`

提供工具：
- `get_sdk_for_vendor(vendor)` — 查询厂商对应的 SDK
- `get_programming_guide(sdk)` — 获取编程指南（语法、编译器、线程模型）
- `get_tiling_pattern(sdk)` — 获取分块代码模板

### 4.4 OperatorRegistryMCPServer — 算子仓库接口

**文件**: `mcp_servers/operator_registry_server.py`

提供工具：
- `lookup(op_name, gpu_model)` — 查找算子
- `register(entry)` — 注册新算子
- `find_similar(op_name, gpu_model)` — 模糊查找相似算子

### 4.5 RemoteExecutorMCPServer — 远程编译/测试

**文件**: `mcp_servers/remote_executor_server.py`

提供工具：
- `compile_kernel(source_code, sdk, build_flags)` — 编译内核代码
- `run_correctness_test(...)` — 运行正确性测试
- `run_benchmark(...)` — 运行性能基准测试

支持三种执行环境：本地 / Docker / SSH 远程。

**重大改进 — 真实的编译和测试实现**：

旧版的 `_run_correctness_test` 和 `_run_benchmark` 是模拟实现（返回固定值），新版是真正在硬件上运行的：

| 后端 | `_run_correctness_test` 实现 | `_run_benchmark` 实现 |
|------|-----|-----|
| **CUDA** | 编译为 `.so` → ctypes 加载 → 与 PyTorch 参考实现对比 | PyTorch 参考函数在 CUDA 设备上计时（warmup + benchmark） |
| **AscendC** | `torch_npu` 参考实现 sanity check | `torch.npu.synchronize()` + 计时 |
| **Triton** | `exec()` 运行 Triton 代码 + `torch` 对比 | PyTorch 参考在 CUDA 设备上计时 |
| 无硬件 | 返回 `status: "skipped"` | 返回 `status: "skipped"` |

**安全修复 — SSH 改用 scp**：

旧版通过 `echo "source_code" | ssh host` 上传代码，存在命令注入风险（代码中的特殊字符可能被 shell 解释）。新版改为：

```
1. 将源代码写入本地临时文件
2. 用 scp 上传到远程机器
3. ssh 执行编译命令
4. 清理本地临时文件
```

---

## 五、工具层 — 基础设施

**目录**: `tools/`

### 5.1 LLM 客户端

**文件**: `tools/llm_client.py`

统一封装了 4 种 LLM 后端：

| 后端 | 说明 |
|------|------|
| `QwenClient` | 阿里云 Qwen3-235B（推荐，通过 DashScope API，**用 httpx 直接调用，不依赖 openai 包**） |
| `OpenAIClient` | OpenAI GPT-4 系列（支持自定义 base_url） |
| `AnthropicClient` | Anthropic Claude 系列 |
| `MockLLMClient` | 模拟客户端（不调用真实 API，用于测试） |

调用方式统一：`await client.chat(system=..., user=..., temperature=0.1)`

**新增 — LLMCache（SQLite 响应缓存）**：

```
BaseLLMClient
├── chat()           → 带缓存的公开接口
│   ├── 查 cache → 命中则直接返回
│   ├── 未命中 → 调用 _raw_chat()（子类实现的真实 API 调用）
│   └── 将响应写入 cache
└── _raw_chat()      → 各子类实现的真实 API 调用（abstract）
```

**LLMCache 技术细节**：
- **存储**：SQLite（`.llm_cache.db`），WAL 模式保证并发安全
- **缓存 key**：`SHA256(model + system_prompt + user_prompt + temperature)` 的前 16 位（content-addressed）
- **过期**：默认 7 天 TTL，自动清理
- **接口**：`get()` / `put()` / `stats()` / `clear()`

**为什么需要缓存**：相同算子 + 相同 GPU + 相同 prompt → LLM 的输出几乎相同。缓存避免了重复调用 API，既省钱又省时间（尤其在 ReviewLoop 多轮迭代时）。

### 5.2 CPU 模拟器

**文件**: `tools/cpu_simulator.py`

没有 GPU 时，用 PyTorch CPU 验证算子的数学正确性（不测硬件性能、只测数学对不对）。

**前向参考实现**（`REFERENCE_IMPLS`）：内置 20+ 种算子的 PyTorch CPU 实现（gelu、silu、rmsnorm、flash_attention、matmul、softmax、layernorm、swiglu 等），能自动生成测试数据并对比。

**新增 — 反向参考实现**（`BACKWARD_REFERENCE_IMPLS`）：

```python
BACKWARD_REFERENCE_IMPLS = {
    "gelu":    _backward_gelu,     # GELU 梯度
    "silu":    _backward_silu,     # SiLU 梯度
    "softmax": _backward_softmax,  # Softmax 梯度
    "rmsnorm": _backward_rmsnorm,  # RMSNorm 梯度
    "matmul":  _backward_matmul,   # MatMul 梯度
}
```

**新增 — `verify_backward()` 方法**：

```python
def verify_backward(self, operator_name: str, rtol: float = 1e-2) -> SimulationResult:
    """用 torch.autograd.gradcheck 验证 backward 正确性"""
```

使用 PyTorch 的有限差分法（gradcheck）自动验证解析梯度是否正确 —— 这是数值验证 backward 实现的黄金标准方法。

### 5.3 模型路由器

**文件**: `tools/model_router.py`

根据任务复杂度选择不同的 LLM：
- **简单算子**（gelu、silu、relu 等逐元素操作）→ 快速模型（如 qwen-plus），省钱快速
- **复杂算子**（flash_attention、matmul、fused_moe）→ 强模型（如 qwen3-235b），保证质量

---

## 六、Prompt 层 — 与 LLM 对话的艺术

**目录**: `prompts/`

### code_gen_prompts.py

这是代码质量的核心。为每种后端精心设计了 prompt 模板。

**CUDA prompt 包含**：
1. GPU 规格（SM 数量、Shared Memory 大小、Compute Capability）
2. 算子数学定义 + 参考实现
3. 编译约束（"不要用 `__float22half2_rn`，改用 `__floats2half2_rn`"）
4. 编译错误知识库动态生成的禁用列表
5. 从算子仓库检索的历史成功代码（Few-shot 示例）
6. 优化建议

**HIP prompt** — **新增禁用 API 列表**（`_HIP_FORBIDDEN`）：
```
- hipBfloat16ToFloat / hipFloatToBfloat16：不存在
- __shfl_sync：HIP 使用 __shfl（无 _sync 后缀）
- wavefront 是 64 线程（不是 CUDA 的 32）
...
```

**AscendC prompt** — **新增禁用 API 列表**（`_ASCENDC_FORBIDDEN`）+ **Few-shot 支持**：
```
- 不要直接用 GM 指针做计算，必须先 DataCopy 到 UB
- 不要使用 Sigmoid / Tanh / Gelu 等高级 API（AscendC 没有内置这些）
- DataCopy 的 count 必须 32 字节对齐
- Cube 矩阵乘法 M/K/N 维度必须是 16 的倍数
...
```
旧版 AscendC prompt 没有 few-shot，新版和 CUDA 一样从算子仓库检索历史成功代码注入。

**Few-shot 注入改进**（`_get_fewshot_example()`）：
- 收集精确匹配 + 相似匹配的候选
- 按 `verification_level`（验证等级）+ `bandwidth_utilization`（带宽利用率）双维度排序，选最佳
- 截断到合理长度（3000 字符）避免 prompt 过长
- 显示验证等级和带宽利用率，给 LLM 参考价值信号

**新增 — 反向传播 prompt**：

| 函数 | 用途 |
|------|------|
| `build_cuda_backward_prompt(op_ir, gpu_spec, forward_code)` | CUDA backward kernel 生成 prompt |
| `build_ascendc_backward_prompt(op_ir, gpu_spec, forward_code)` | AscendC backward kernel 生成 prompt |

Backward prompt 包含：
- 反向传播数学定义（`backward_math_description`）
- `saved_for_backward` 中的张量列表
- backward 参考实现（PyTorch 代码）
- 如果有 forward kernel 代码，也会注入以确保数学一致性

**其他 prompt**：
- `build_cuda_simple_prompt()` — 降级简单版（编译失败多次后使用）
- `build_optimization_prompt()` — 性能优化专用

---

## 七、后端层 — 硬件适配

**目录**: `backends/`

### backends/ascend/ascendc_codegen.py

昇腾 AscendC 专用的代码生成器，包含：
- CUDA → AscendC 概念映射表
- Tiling 配置自动计算（基于 UB/L0A/L0B 大小约束）
- 模板代码生成

---

## 八、入口层 — 用户接口

### 8.1 cli.py — 命令行工具

基于 Click 框架，提供：

| 命令 | 功能 |
|------|------|
| `generate "..."` | 自然语言生成算子 |
| `generate --op X --gpu Y` | 参数化生成 |
| `generate --review/--no-review` | **新增**：控制是否运行 ReviewLoop 验证（默认开启） |
| `npu-test` | 昇腾 NPU 测试 |
| `registry list/show/stats/history` | 算子仓库管理 |
| `registry search` | **新增**：多条件组合搜索（--op, --gpu, --backend, --min-bw, --verified-only） |
| `registry export` | **新增**：导出算子仓库为 JSON 文件 |
| `kb stats` | **新增**：查看编译错误知识库统计 |
| `kb export` | **新增**：导出知识库（可提交 git 分享给团队） |
| `kb import` | **新增**：从 JSON 文件导入知识库 |
| `cache stats` | **新增**：查看 LLM 缓存统计 |
| `cache clear` | **新增**：清空 LLM 缓存 |
| `interactive` | 交互模式（支持多轮追问） |
| `interactive --no-review` | **新增**：交互模式中关闭 ReviewLoop |

**CLI 的 `generate` 命令现在默认启用 `--review`**（运行完整的 ReviewLoop 五阶段验证），加 `--no-review` 可跳过。

### 8.2 main.py — Python API

提供 `run_operator_generation()` 异步函数，适合编程调用。自动构建 Agent 系统 → 执行工作流 → 返回结果。

**改进 — 完整读取 config.yaml**：

```python
# 旧版只读 backend，新版读取所有 LLM 参数
llm_kwargs = {}
if llm_cfg.get("model"):            llm_kwargs["model"] = llm_cfg["model"]
if llm_cfg.get("base_url"):         llm_kwargs["base_url"] = llm_cfg["base_url"]
if llm_cfg.get("api_key"):          llm_kwargs["api_key"] = llm_cfg["api_key"]
if "enable_thinking" in llm_cfg:    llm_kwargs["enable_thinking"] = llm_cfg["enable_thinking"]

llm_client = create_llm_client(backend=llm_backend, **llm_kwargs)
```

这意味着 config.yaml 中可以配置：
```yaml
llm:
  backend: qwen
  model: qwen3-235b-a22b
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  api_key: sk-xxx
  enable_thinking: false
```

**Qwen 后端正确支持**：`QwenClient` 使用 httpx 直接调用 DashScope OpenAI 兼容接口，不再依赖 openai 包的版本兼容性。

### 8.3 train.py — 训练入口

**改进 — `--llm` 参数支持 qwen**：

```bash
python train.py --script my_train.py --gpus h100_sxm5 --llm qwen
```

支持的 LLM 后端：`qwen`、`openai`、`anthropic`、`mock`（默认 mock）。

### 8.4 orchestrator_v2.py — V2 主调度器

**双路径编排**：
- **Path A**（已知 GPU）：查算子仓库 → 直接复用 → 跳过生成
- **Path B**（未知 GPU）：GPU 发现 → SDK 解析 → 代码生成 → Review Loop → 存入仓库

---

## 九、配置与存储

| 文件 | 用途 |
|------|------|
| `config/config.yaml` | 全局配置（LLM 后端 + model/base_url/api_key/enable_thinking、优化参数、验证阈值、工作流设置） |
| `.env` | API 密钥（QWEN_API_KEY / OPENAI_API_KEY / ANTHROPIC_AUTH_TOKEN） |
| `.operator_registry.db` | SQLite 算子仓库（版本管理 + 并发安全 + WAL 模式 + verification_level 字段） |
| `.compile_errors.json` | 编译错误知识库（36 条规则 + 动态学习 + 导入导出） |
| `.llm_cache.db` | **新增**：LLM 响应缓存（SQLite, content-addressed, 7 天 TTL） |
| `.review_checkpoints/` | **新增**：ReviewLoop 断点文件（JSON，1 小时过期） |
| `requirements.txt` | 依赖清单（**新增 httpx>=0.27.0**，用于 Qwen DashScope HTTP 客户端） |

---

## 十、测试体系

| 目录 | 类型 | 说明 |
|------|------|------|
| `tests/unit/` | 单元测试 | 单个组件功能测试（HardwareProfiler、SpecAnalyzer 等） |
| `tests/simulation/` | 模拟测试 | 无 GPU 完整流程模拟（7 个部分，用 CPU 模拟器 + Mock LLM） |
| `tests/hetero/hetero_test.py` | 异构测试 | CUDA GPU 三阶段测试（Phase 0/1/2） |
| `tests/hetero/npu_test.py` | NPU 测试 | 昇腾 910B 专用测试框架（AscendC + torch_npu） |
| `tests/hetero/run_phase1_npu.sh` | Shell 脚本 | NPU 测试一键脚本 |
| `tests/hetero/run_phase1_slurm.sh` | Slurm 脚本 | 集群提交脚本（支持 fnlp-4090 分区） |
