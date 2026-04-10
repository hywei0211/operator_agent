# Agent 完整工作流程

> 本文档用一个**具体例子**，走一遍系统从用户输入到最终输出的完整流程。  
> 每一步都标注了对应的代码文件和函数，方便对照阅读。

---

## 目录

- [场景设定](#场景设定)
- [流程总览图](#流程总览图)
- [Phase 0: 用户输入 & 意图解析](#phase-0-用户输入--意图解析)
- [Phase 1: 训练代码分析](#phase-1-训练代码分析)
- [Phase 2: 硬件分析 & GPU 发现](#phase-2-硬件分析--gpu-发现)
- [Phase 3: SDK 解析](#phase-3-sdk-解析)
- [Phase 4: 算子规格解析](#phase-4-算子规格解析)
- [Phase 5: Tiling 分块计算](#phase-5-tiling-分块计算)
- [Phase 6: Forward 代码生成](#phase-6-forward-代码生成)
- [Phase 7: Backward 代码生成](#phase-7-backward-代码生成)
- [Phase 8: Review Loop — 六级硬件自适应验证](#phase-8-review-loop--六级硬件自适应验证)
- [Phase 9: 分布式部署方案](#phase-9-分布式部署方案)
- [Phase 10: 训练执行 & 监控](#phase-10-训练执行--监控)
- [两种调度器的区别](#两种调度器的区别)
- [错误自修复流程详解](#错误自修复流程详解)
- [LLM 缓存机制](#llm-缓存机制)
- [数据流转全景图](#数据流转全景图)

---

## 场景设定

假设用户输入：

```bash
python cli.py generate "帮我生成 SiLU 激活函数算子，目标昇腾 910B" --llm qwen --review
```

> `--review` 为默认开启，使用 `--no-review` 可跳过 ReviewLoop 验证阶段（快速迭代用）。

接下来，我们跟踪每一步发生了什么。

---

## 流程总览图

```
用户输入: "帮我生成 SiLU 算子，目标昇腾 910B"
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 0: 意图解析 (IntentParser)                                 │
│ "SiLU 算子" + "昇腾 910B" → {op: silu, gpu: ascend_910b}        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
  ▼                            ▼
┌─────────────────┐   ┌──────────────────────────────────────────┐
│ Phase 1: 训练   │   │ Phase 2: 硬件分析 (HardwareProfiler)      │
│ 代码分析        │   │ ascend_910b → GPUSpec (32 AI Core,        │
│ (如果有训练代码) │   │   256KB UB, DaVinci v2 架构)              │
└─────────────────┘   │ + HardwareDetector 检测本机环境            │
                      └──────────────────────────────┬───────────┘
                                                      │
                      ┌───────────────────────────────▼───────────┐
                      │ Phase 3: SDK 解析 (SDKResolver)            │
                      │ 华为昇腾 → AscendC SDK                     │
                      │ (手动 DataCopy, Cube 16×16, 双缓冲)        │
                      └───────────────────────────────┬───────────┘
                                                      │
                      ┌───────────────────────────────▼───────────┐
                      │ Phase 4: 算子规格解析 (SpecAnalyzer)        │
                      │ "silu" → 匹配模板 → OperatorIR             │
                      │ {math: "y = x * sigmoid(x)", dtype: fp16,  │
                      │  backward: "grad_x = grad_y * ...",        │
                      │  saved_for_backward: ["x"]}                │
                      └───────────────────────────────┬───────────┘
                                                      │
                      ┌───────────────────────────────▼───────────┐
                      │ Phase 5: Tiling 计算 (TilingAgent)         │
                      │ UB=256KB → tile_length=21760               │
                      │ 双缓冲, 128 对齐                            │
                      └───────────────────────────────┬───────────┘
                                                      │
                      ┌───────────────────────────────▼───────────┐
                      │ Phase 6: Forward 代码生成 (CodeGenAgent)    │
                      │ Prompt = 算子定义 + GPU规格 + SDK上下文      │
                      │   + 错误知识库 + 禁用API列表                │
                      │   + Few-shot 示例 (按 verification_level   │
                      │     + bandwidth_utilization 排序选最佳)     │
                      │ → LLM (Qwen3-235B) → AscendC C++ 代码     │
                      │   [缓存命中? → 直接返回, 不调 LLM]          │
                      └───────────────────────────────┬───────────┘
                                                      │
                      ┌───────────────────────────────▼───────────┐
                      │ Phase 7: Backward 代码生成                  │
                      │ CodeGenAgent.generate_backward()            │
                      │ 输入: OperatorIR.backward_math_description  │
                      │       + forward_kernel 代码                 │
                      │ → LLM → backward AscendC C++ 代码          │
                      │ → gradcheck 数值验证                        │
                      └───────────────────────────────┬───────────┘
                                                      │
                      ┌───────────────────────────────▼───────────┐
                      │ Phase 8: Review Loop (6 级硬件自适应验证)    │
                      │ HardwareDetector 自动检测 → 验证深至最高级   │
                      │ ┌──────────────────────────────────────┐   │
                      │ │ Level 1: 静态分析 (总是执行)           │   │
                      │ │ Level 2: LLM 代码审查 (有 LLM 时)     │   │
                      │ │ Level 3: CPU 数学验证 (有 PyTorch 时)  │   │
                      │ │ Level 4: 真实编译 (有编译器时)          │   │
                      │ │ Level 5: 硬件运行+数值验证 (有GPU/NPU) │   │
                      │ │ Level 6: 硬件 Benchmark (有GPU/NPU)   │   │
                      │ └──────────────────────────────────────┘   │
                      │ checkpoint/resume: 每 stage 存盘，          │
                      │   crash 后从上次位置继续                     │
                      │ progress callback: 实时通知 CLI 进度        │
                      │ 失败 → CodeGen/Optimizer 修复 → 循环        │
                      │ 通过 → 写入算子仓库 (SQLite)                │
                      └───────────────────────────────┬───────────┘
                                                      │
                                                      ▼
                                              ✅ 输出结果
                                         AscendC C++ 源代码 (forward + backward)
                                         保存到 output/ 目录
                                         写入算子仓库 (SQLite)
```

---

## Phase 0: 用户输入 & 意图解析

**代码入口**: `cli.py` → `_parse_with_clarification()` → `IntentParser.parse()`

```
用户输入: "帮我生成 SiLU 激活函数算子，目标昇腾 910B"
```

### 步骤 1: CLI 接收输入

`cli.py` 的 `generate` 命令检测到用户传入了自然语言（没有 `--op` 参数），进入 LLM 意图解析模式。

CLI 新增的参数：
```bash
# 默认开启 ReviewLoop 验证
python cli.py generate "生成 SiLU 算子" --review        # 显式开启（默认）
python cli.py generate --op silu --gpu rtx_4090 --no-review  # 跳过验证，快速迭代

# 新增 CLI 命令
python cli.py registry search --gpu h100_sxm5 --min-bw 0.6  # 搜索仓库
python cli.py kb export ./errors.json                        # 导出知识库
python cli.py kb import ./errors.json                        # 导入知识库
python cli.py cache stats                                    # LLM 缓存统计
python cli.py cache clear                                    # 清空缓存
```

### 步骤 2: IntentParser 调用 LLM

`IntentParser` 构造 system prompt（包含支持的 GPU 列表），将用户输入发送给 Qwen3-235B：

```
System: 你是一个算子生成系统的意图解析器...支持的GPU: ascend_910b, rtx_4090, h100_sxm5...
User:   用户输入：帮我生成 SiLU 激活函数算子，目标昇腾 910B
```

> **LLM 缓存**: 如果相同的 system + user + model + temperature 之前调用过，`LLMCache` 会直接返回缓存结果（SQLite 存储，7 天过期），不消耗 LLM 调用。

### 步骤 3: LLM 返回结构化结果

```json
{
  "status": "ready",
  "operator": "silu",
  "gpus": ["ascend_910b"],
  "backend": "ascendc",
  "dtype": "fp16",
  "confidence": 0.95
}
```

status = "ready" 表示信息完整，不需要追问。如果用户只说了"写个 RoPE"（没说目标硬件），status 会是 "need_clarification"，系统会追问。

### 步骤 4: 开始生成

CLI 调用 `_do_generate(op_name="silu", gpus=["ascend_910b"], backend="ascendc", review=True)`

---

## Phase 1: 训练代码分析

**代码**: `agents/training_analyst.py` → `TrainingAnalystAgent.run()`

> 注意：在 CLI 的 `generate` 模式下，这一步可能跳过（因为用户直接指定了算子）。
> 在 V2 Orchestrator 的完整流程中（如从训练代码出发），这一步会执行。

如果有训练代码输入，分析器会：
1. 用正则扫描代码中的 `F.silu`、`SiLU` 等关键词 → 识别出 silu 算子
2. 检测模型架构（如识别到 LlamaRMSNorm → LLaMA 架构）
3. 自动补全 LLaMA 架构需要的全部算子：`[flash_attention, rmsnorm, silu, matmul, embedding, softmax]`
4. 按优先级排序：critical（matmul, flash_attention）> required（silu, embedding）> optional（dropout）

---

## Phase 2: 硬件分析 & GPU 发现

**代码**: `agents/hardware_profiler.py` → `HardwareProfilerAgent.run()`  
**硬件检测**: `agents/verifier.py` → `HardwareDetector.detect()`

### 步骤 1: 查找 GPU 规格

收到 `gpu_id = "ascend_910b"`，从 GPU 数据库 (`knowledge_base/hardware_specs/gpu_database.py`) 查找：

```python
GPUSpec(
    model_name="Ascend 910B",
    vendor=GPUVendor.HUAWEI,
    architecture="DaVinci v2",
    compute_units=32,              # 32 个 AI Core
    memory=MemorySpec(
        capacity_gb=64,
        bandwidth_gbps=1600,       # HBM2e
    ),
    compute=ComputeSpec(
        fp16_tflops=320.0,         # FP16 算力
    ),
    supported_backends=[GPUBackend.ASCENDC],
)
```

### 步骤 2: HardwareDetector 检测本机环境

`HardwareDetector` 是 6 级验证的基础，它一次性检测当前机器的所有硬件能力（结果缓存，全进程只检测一次）：

```python
hw = HardwareDetector.detect()
# → {
#     "nvidia_gpu": False, "nvidia_gpu_name": "",
#     "nvcc": False,
#     "amd_gpu": False, "hipcc": False,
#     "npu": True, "npu_name": "Ascend 910B",   # ← 检测到昇腾 NPU
#     "cann": True,                               # ← 检测到 CANN 编译器
#     "torch": True,                              # ← 检测到 PyTorch
# }
```

检测方法：
- **NVIDIA GPU**: `torch.cuda.is_available()` + `torch.cuda.get_device_name()`
- **昇腾 NPU**: `import torch_npu` → `torch.npu.is_available()`
- **AMD GPU**: `rocm-smi --showid` 命令
- **编译器**: `nvcc --version` / `hipcc --version` / `atc --version`（均通过 subprocess 5 秒超时检测）

这一步决定了后续 Review Loop 能验到哪一级。

### 步骤 3: 存入共享上下文

```python
context.add_artifact("hardware_profiles", {"ascend_910b": gpu_spec})
```

---

## Phase 3: SDK 解析

**代码**: `agents/sdk_resolver.py` → `SDKResolverAgent.run()`

根据 `vendor = HUAWEI`，确定使用 AscendC SDK，构建代码生成上下文：

```python
SDKContext(
    sdk_name="ascendc",
    language="C++",
    compiler="ccec (CANN Compiler)",
    kernel_decorator="__aicore__",
    thread_id_expr="GetBlockIdx()",
    shared_memory_syntax="TBuf<QuePosition::VECIN>",
    sync_primitive="pipe.Barrier()",
    memory_management="manual",      # ← 关键：昇腾必须手动搬运数据
    tiling_pattern="...",            # 标准分块代码模板
    extra_notes="""
        关键: 必须手动 DataCopy，不能直接访问 GM 指针做计算
        矩阵分块固定为 16×16 的倍数（Cube 单元硬性约束）
        双缓冲是标配，否则 MTE 搬运会成为瓶颈
    """,
)
```

---

## Phase 4: 算子规格解析

**代码**: `agents/spec_analyzer.py` → `OperatorSpecAgent.run()`

### 步骤 1: 模板匹配

"silu" 匹配到内置模板 `OPERATOR_TEMPLATES["silu"]`，直接使用（不需要调用 LLM）。

### 步骤 2: 生成 OperatorIR（含 backward 字段）

`OperatorIR` 现在包含了**反向传播**相关字段：

```python
OperatorIR(
    name="silu",
    category=OperatorCategory.ELEMENTWISE,
    description="Sigmoid Linear Unit (SiLU/Swish) activation",
    inputs=[TensorSpec(name="x", shape=["batch","seq_len","hidden"], dtype=FP16)],
    outputs=[TensorSpec(name="y", shape=["batch","seq_len","hidden"], dtype=FP16)],

    # Forward 数学
    math_description="y = x * sigmoid(x) = x / (1 + exp(-x))",
    reference_impl="torch.nn.functional.silu(x)",

    # Backward 数学 (新增)
    backward_math_description="grad_x = grad_y * sigmoid(x) * (1 + x * (1 - sigmoid(x)))",
    backward_reference_impl="...",  # PyTorch backward 参考实现
    saved_for_backward=["x"],       # forward 中需保存供 backward 使用的张量

    # 复杂度分析
    flops_formula="4 * batch * seq_len * hidden",
    memory_reads_formula="batch * seq_len * hidden",
    memory_writes_formula="batch * seq_len * hidden",
    tags=["activation", "swish", "llama", "qwen"],
)
```

---

## Phase 5: Tiling 分块计算

**代码**: `agents/tiling_agent.py` → `TilingAgent._tiling_for_ascend()`

SiLU 是逐元素操作（elementwise），计算逻辑：

```
UB 总容量 = 256 KB
双缓冲 → 有效 UB = 128 KB
FP16 每元素 2 字节，输入+输出共 3 个 tensor（x, sigmoid(x), y）
每次可处理 = 128KB / (3 × 2B) = 21,845 个元素
对齐到 128 → tile_length = 21,760
```

输出：
```python
TilingConfig(
    recommended={"tile_length": 21760, "double_buffer": True},
    constraints={"ub_kb": 256, "vector_width": 128, "memory_model": "manual"},
    estimated_sram_utilization=0.75,
)
```

---

## Phase 6: Forward 代码生成

**代码**: `agents/code_generator.py` → `CodeGenAgent._generate_ascendc()`

### 步骤 1: 构造 Prompt

系统调用 `build_ascendc_codegen_prompt(op_ir, gpu_spec)` 构造精心设计的 prompt，包含：

```
[1] 算子数学定义: y = x * sigmoid(x)
[2] GPU 规格: 昇腾 910B, 32 AI Core, UB 256KB, DaVinci v2
[3] AscendC 编程规范: 必须用 DataCopy, 不能直接访问 GM, 双缓冲...
[4] 禁用 API 列表 (AscendC): 不要直接访问 GM 指针, 不要用不存在的 AscendC::Sigmoid...
[5] 编译错误知识库: 从 KB 动态生成的 "不要用 xxx 函数，改用 yyy" 片段
[6] Few-shot 示例: 从仓库检索的历史成功代码
[7] 输出格式: JSON {kernel_code, build_flags, optimizations}
```

### Few-shot 示例选择（改进）

`_get_fewshot_example()` 现在按 **verification_level + bandwidth_utilization 双维度排序**，选最佳：

```python
# 从仓库收集候选：精确匹配（同算子+同 GPU）和相似匹配（同算子其他 GPU）
candidates.sort(key=lambda e: (
    _level_rank[e.verification_level],  # 优先验证等级高的（如 benchmarked > compiled）
    e.bandwidth_utilization,             # 同等级选带宽利用率最高的
), reverse=True)
best = candidates[0]
```

### 禁用 API 列表（三后端全覆盖）

系统为 **CUDA、HIP、AscendC** 三个后端都维护了禁用 API 列表，防止 LLM 生成不存在的 API：

| 后端 | 禁用示例 |
|------|---------|
| **CUDA** | `__float22half2_rn` 签名错误，用 `__floats2half2_rn`；不要用 `Sigmoid` 头文件 |
| **HIP** | `hipBfloat16ToFloat` 不存在，用 `__bfloat162float`；不要用 CUDA 专属的 `__shfl_sync` |
| **AscendC** | 不要直接用 GM 指针计算；不要 `AscendC::Sigmoid`（不存在）；必须手动 DataCopy |

禁用列表有两个来源：
1. **硬编码规则**（`_CUDA_FORBIDDEN_FALLBACK`, `_HIP_FORBIDDEN`, `_ASCENDC_FORBIDDEN`）
2. **编译错误知识库动态生成**（`_get_backend_forbidden(backend)`）——随着使用越来越智能

### 步骤 2: 调用 LLM（带缓存）

```python
response = await self.call_llm(prompt, max_tokens=8192)
```

**LLM 缓存机制**：`LLMCache` 会先查 SQLite 缓存（key = hash(model + system + user + temperature)）。缓存命中则直接返回，不消耗 API 调用。

Qwen3-235B 生成约 193 行 AscendC C++ 代码，包含：
- 双缓冲流水线
- Vector 指令向量化
- 32 AI Core 均匀分块
- 16 字节内存对齐

### 步骤 3: 解析响应

从 LLM 响应中提取代码（支持 JSON 格式、markdown 代码块、裸代码等多种格式）。

### 步骤 4: 输出 GeneratedKernel

```python
GeneratedKernel(
    operator_name="silu",
    backend="ascendc",
    target_gpu="Ascend 910B",
    source_code="// 193 lines of AscendC C++ ...",
    build_flags=["--target=ascend910b", "-O2"],
)
```

---

## Phase 7: Backward 代码生成

**代码**: `agents/code_generator.py` → `CodeGenAgent.generate_backward()`

> 这是新增的阶段。当 `OperatorIR.backward_math_description` 存在时，系统会自动生成 backward kernel。

### 步骤 1: 检查 backward 定义

```python
if op_ir.backward_math_description:
    # "grad_x = grad_y * sigmoid(x) * (1 + x * (1 - sigmoid(x)))"
    # saved_for_backward = ["x"]  → backward 需要 forward 保存的 x
```

### 步骤 2: 构造 backward prompt

调用 `build_ascendc_backward_prompt(op_ir, gpu_spec, forward_code)`，包含：
- backward 数学公式
- forward kernel 代码（供参考风格和数据布局）
- `saved_for_backward` 列表（backward 需要的 forward 输入）
- 后端对应的禁用 API 列表

### 步骤 3: LLM 生成 backward kernel

```python
backward_result = await codegen.generate_backward(
    ctx,
    operator_ir=op_ir,
    gpu_spec=gpu_spec,
    forward_kernel=forward_kernel,
)
backward_kernel = backward_result.output
# → GeneratedKernel(operator_name="silu_backward", backend="ascendc", ...)
```

### 步骤 4: gradcheck 验证

通过 `CPUSimulator.verify_backward()` 在 CPU 上用 PyTorch autograd 数值 gradcheck 验证 backward 正确性：

```python
# CPU 数学验证（VerifierAgent Level 3）
backward_result = sim.verify_backward("silu")
# 用 torch.autograd.gradcheck 对比数值梯度 vs 解析梯度
```

---

## Phase 8: Review Loop — 六级硬件自适应验证

**代码**: `agents/review_loop.py` → `ReviewLoopAgent.run()`  
**验证器**: `agents/verifier.py` → `VerifierAgent.run()`

这是整个系统**最核心的质量控制环节**。最多迭代 5 轮。

### 六级验证体系

验证器 `VerifierAgent` 基于 `HardwareDetector` 的检测结果，**自动决定能验到哪一级**：

```
Level 1: 静态分析        ← 总是执行（零依赖）
Level 2: LLM 代码审查    ← 有 LLM 时执行
Level 3: CPU 数学验证    ← 有 PyTorch 时执行（forward + backward）
Level 4: 真实编译        ← 有匹配编译器时执行（nvcc/hipcc/cann）
Level 5: 硬件运行+数值   ← 有匹配 GPU/NPU 时执行
Level 6: 硬件 Benchmark  ← 有匹配 GPU/NPU 时执行
```

每一级包含前面所有级的检查，任何一级失败都会停下来并反馈修复建议。

| 环境 | 最高可达等级 |
|------|------------|
| 纯 CPU 无 PyTorch | Level 1 (静态分析) |
| 有 PyTorch 无 GPU | Level 3 (CPU 数学验证) |
| 有 nvcc 无 NVIDIA GPU | Level 4 (编译通过) |
| 有昇腾 NPU + CANN | Level 6 (完整 Benchmark) |

### checkpoint/resume 机制

```python
# 每个 stage 完成后自动保存断点到 .review_checkpoints/
self._save_checkpoint(op_name, gpu_model, iteration, current_code, history, stages)

# 下次运行时自动恢复（1 小时内有效）
ckpt = self._load_checkpoint(op_ir.name, gpu_spec.model_name)
if ckpt:
    iteration = ckpt["iteration"]
    current_kernel = ... # 从断点恢复代码
```

这意味着：如果验证过程中 crash（网络超时、进程被 kill 等），重新运行会**自动跳过已通过的阶段**。

### progress callback（实时进度通知）

CLI 通过注入回调函数实时显示验证进度：

```python
review_agent.set_progress_callback(_on_stage)

# CLI 输出效果：
#   [1/5] ⏳ 静态审查...
#   [1/5] ✅ 静态审查
#   [1/5] ⏳ 编译检查...
#   [1/5] ❌ 编译检查 — 修复中...
#   [2/5] ⏳ 静态审查...
#   ...
```

### 第 1 轮迭代

#### Stage 1: 静态代码审查 `_stage_static_review()`

检查项：
- ✅ 代码长度 > 50 字符
- ✅ 大括号匹配
- ✅ 数学逻辑正确（LLM 检查 sigmoid 实现）
- ✅ 边界检查存在

→ **通过**，进入下一阶段

#### Stage 2: 编译检查 `_stage_compile()`

通过 MCP 调用 `RemoteExecutorServer.compile_kernel()`：
```
ccec --target=ascend910b -O2 silu_kernel.cpp
```

**假设第 1 次编译失败**：
```
error: 'Sigmoid' is not a member of 'AscendC'
```

→ **失败**，记录错误到知识库，保存 checkpoint，反馈给 CodeGenAgent 重写

#### 修复过程

```python
fix_context = {
    "issues_to_fix": ["'Sigmoid' is not a member of 'AscendC'"],
    "fix_guidance": "Use Div + Exp + Add to implement sigmoid manually",
    "history_summary": "Iter 1, compile: 'Sigmoid' not found",
}
```

CodeGenAgent 收到修复上下文，重新生成代码（这次知道不能用 `Sigmoid` 函数）。

### 第 2 轮迭代

#### Stage 1: 静态审查 → 通过
#### Stage 2: 编译检查 → **通过**（修复有效）

#### Stage 3: 正确性验证 `_stage_correctness()`

`VerifierAgent` 分层执行：

```
Level 3 (CPU 数学): 用 PyTorch CPU 参考实现验证
  - forward: torch.nn.functional.silu(x) vs 生成代码
  - backward: gradcheck (如果 OperatorIR 有 backward 定义)

Level 5 (硬件数值): 在真实 NPU 上运行
  expected = torch.nn.functional.silu(x)
  actual = run_ascendc_kernel(x)
  relative_error = max(|expected - actual| / |expected|)
```

结果：`error = 0.00e+00`，`verification_level = hw_verified` → **通过**

#### Stage 4: 性能基准 `_stage_performance()`

```
Level 6 (硬件 Benchmark): 在真实 NPU 上跑 benchmark
  bandwidth_utilization = 实际吞吐 / 理论峰值带宽
```

结果：`bandwidth_utilization = 0.72` (72%) ≥ 目标 0.55 (55%)，`verification_level = benchmarked` → **通过**

> 注意：如果没有真实硬件 benchmark 数据（Level < 6），性能判定阈值会自动降低到 `min_bw_efficiency × 0.6`。

#### Stage 5: 综合评审 `_stage_meta_review()`

所有前 4 个阶段都通过，平均分 > 0.7 → **最终通过** ✅

### 写入算子仓库

```python
OperatorEntry(
    operator_name="silu",
    gpu_model="Ascend 910B",
    backend="ascendc",
    source_code="...",
    correctness_passed=True,
    bandwidth_utilization=0.72,
    iteration_count=2,
    verification_level="benchmarked",    # 新增：记录验证等级
    version=1,
)
```

存入 SQLite 数据库 `.operator_registry.db`。

---

## Phase 9: 分布式部署方案

**代码**: `agents/distribution.py` → `DistributionAgent.run()`

> 本例只有单个 GPU，此阶段较简单。多 GPU 异构场景下这一步很重要。

如果是异构集群（如 H100 + MI300X）：
```
H100 FP16: 989 TFLOPs  → 负载权重 43%
MI300X FP16: 1307 TFLOPs → 负载权重 57%
通信后端: UCC (异构集群)
通信模式: AllReduce
```

---

## Phase 10: 训练执行 & 监控

**代码**: `agents/training_executor.py` + `agents/runtime_monitor.py`

> 在 CLI `generate` 模式下，这两步通常不执行（只生成代码，不启动训练）。
> 在 V2 Orchestrator 的完整流程中会执行。

### TrainingExecutor 做的事：
1. 加载已验证的 silu 算子代码（forward + backward）
2. 生成 `torch.library` 注册代码（自动加载 .so）
3. **生成 `torch.autograd.Function` 包装**（新增）：
   ```python
   class SiLUCustomOp(torch.autograd.Function):
       @staticmethod
       def forward(ctx, x):
           ctx.save_for_backward(x)
           return custom_silu_forward(x)

       @staticmethod
       def backward(ctx, grad_output):
           x, = ctx.saved_tensors
           return custom_silu_backward(grad_output, x)
   ```
   这一步由 `_generate_autograd_wrappers()` 自动完成——如果 kernel 有 backward_source_code，生成完整的 forward+backward Function；否则只生成 forward 包装。
4. 修改训练脚本，替换 `F.silu` 调用
5. 生成 `torchrun --nproc_per_node=8 train.py` 启动命令

### RuntimeMonitor 做的事：
每 30 秒轮询一次：
- GPU 利用率 < 70% → 警告"利用率过低，考虑增大 batch size"
- 显存 > 90% → 警告"显存即将耗尽，考虑开启梯度检查点"
- Loss 100 步没下降 → 警告"训练可能停滞"

---

## 两种调度器的区别

系统有两个调度器版本：

### V1 Orchestrator (`agents/orchestrator.py`)

```
串行流水线：
硬件分析 → 算子解析 → 代码生成(forward+backward) → 优化迭代 → 验证 → 分布式部署
```

- 适合简单场景（单算子、已知 GPU）
- `main.py` 使用的是 V1

### V2 MasterOrchestrator (`orchestrator_v2.py`)

```
双路径：
GPU 列表 → 分类已知/未知
  ├─ 已知 GPU → 查仓库 → 直接复用（不重新生成！）
  └─ 未知 GPU → GPU发现 → SDK解析 → 并行生成(forward+backward) → ReviewLoop → 存仓库
最后合并 → 分布式策略 → 训练执行(含 autograd wrapper 生成) → 监控
```

- 适合完整场景（训练代码分析、多 GPU、混合已知/未知）
- 支持**并行生成**：多个算子 × 多种 GPU 同时生成
- 支持**仓库复用**：已验证的算子不重新生成
- `cli.py` 的 `generate` 命令使用的是简化版（直接调 SpecAnalyzer + CodeGen）

---

## 错误自修复流程详解

这是系统最有价值的设计之一。以编译错误为例（**CUDA、HIP、AscendC 三后端都有对应的禁用 API 列表和自动修复规则**）：

```
                    ┌─────────────────┐
                    │   CodeGenAgent   │
                    │  生成 kernel 代码 │
                    │ (CUDA/HIP/AscendC)│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  编译 (nvcc /    │
                    │  hipcc / ccec)   │
                    └────────┬────────┘
                             │
                     ┌───────▼───────┐
                     │  编译成功？     │
                     └───┬───────┬───┘
                    Yes  │       │ No
                         │       │
                    ┌────▼──┐  ┌─▼──────────────────────┐
                    │ 继续   │  │ 1. 记录错误到知识库      │
                    │ 验证   │  │ 2. 尝试 auto_fix         │
                    └───────┘  │    (36条自动修复规则)     │
                               │ 3. 重新编译               │
                               └──────────┬───────────────┘
                                          │
                                   ┌──────▼──────┐
                                   │ 修复成功？    │
                                   └──┬───────┬──┘
                                 Yes  │       │ No
                                      │       │
                                 ┌────▼──┐  ┌─▼─────────────────────┐
                                 │ 继续   │  │ 反馈给 CodeGenAgent    │
                                 │ 验证   │  │ 附带: 错误信息 +       │
                                 └───────┘  │       历史失败记录 +    │
                                            │       修复建议 +        │
                                            │       禁用API列表       │
                                            └──────────┬─────────────┘
                                                       │
                                            ┌──────────▼─────────────┐
                                            │ 失败次数 ≥ 2?           │
                                            └──┬───────────────┬─────┘
                                          No   │               │ Yes
                                               │               │
                                          ┌────▼────┐   ┌─────▼──────────┐
                                          │ 正常重新 │   │ 自动降级到       │
                                          │ 生成代码 │   │ "简单模式" prompt │
                                          └─────────┘   │ (不用 half2,     │
                                                        │  不用 shared mem, │
                                                        │  纯 float 精度)   │
                                                        └─────────────────┘
```

### 禁用 API 列表（三后端全覆盖）

| 后端 | 来源 | 禁用示例 |
|------|------|---------|
| **CUDA** | `_CUDA_FORBIDDEN_FALLBACK` + KB 动态 | `__float22half2_rn` 签名错误 → `__floats2half2_rn` |
| **HIP** | `_HIP_FORBIDDEN` + KB 动态 | `hipBfloat16ToFloat` 不存在 → `__bfloat162float` |
| **AscendC** | `_ASCENDC_FORBIDDEN` + KB 动态 | GM 指针直接计算 → 必须先 DataCopy 到 UB |

**动态部分**：`_get_backend_forbidden(backend)` 从编译错误知识库 `CompileErrorKB` 中读取最新的错误模式，自动生成 prompt 片段。每次编译失败都会被记录，系统**越用越聪明**。

### 36 条 auto_fix 规则示例

| 错误模式 | 自动修复 |
|---------|---------|
| `__float22half2_rn` 未定义 | 替换为 `__floats2half2_rn` |
| `half2 x = make_float2(...)` 类型不匹配 | 替换为 `make_half2(...)` |
| `wmma::...` 未找到 | 添加 `using namespace nvcuda;` |
| VLA (变长数组) 不支持 | 替换为固定大小数组 |

---

## LLM 缓存机制

**代码**: `tools/llm_client.py` → `LLMCache`

所有 LLM 调用都经过 SQLite 缓存层，避免重复调用浪费时间和金钱：

```
LLM 调用请求
    │
    ▼
┌──────────────────────────────────┐
│ LLMCache.get(model, system,      │
│              user, temperature)   │
│                                   │
│ key = SHA256(model||system||      │
│       user||temperature)[:16]     │
└────────┬───────────────┬─────────┘
    命中  │               │ 未命中
         │               │
    ┌────▼────┐    ┌─────▼─────────────┐
    │ 直接返回 │    │ 调用 LLM API       │
    │ 缓存结果 │    │ (Qwen/OpenAI/Claude)│
    └─────────┘    │ → 存入缓存          │
                   │ → 返回结果          │
                   └───────────────────┘
```

- **存储**: SQLite（`.llm_cache.db`），WAL 模式
- **过期**: 默认 7 天 TTL
- **CLI 管理**:
  ```bash
  python cli.py cache stats   # 查看缓存条数和存储路径
  python cli.py cache clear   # 清空全部缓存
  ```

---

## 数据流转全景图

展示一个请求从头到尾，每个 Agent 产出什么数据、传给谁：

```
用户输入 (自然语言)
    │
    ▼
IntentParser
    │ 产出: {operator: "silu", gpus: ["ascend_910b"]}
    │ [LLM 缓存: 命中 → 直接返回, 未命中 → 调 LLM → 存缓存]
    ▼
TrainingAnalyst (可选)
    │ 产出: TrainingPlan {critical_ops: [...], required_ops: [...]}
    ▼
HardwareProfiler + HardwareDetector
    │ 产出: Dict[gpu_id → GPUSpec]
    │        (32 AI Core, 256KB UB, 1600 GB/s HBM 带宽...)
    │ 产出: hw_capabilities {npu=True, cann=True, torch=True}
    │        → 决定验证能到哪一级
    ▼
GPUDiscovery (如果 GPU 未知)
    │ 产出: DiscoveredGPUInfo {normalized_spec, confidence, source}
    ▼
SDKResolver
    │ 产出: Dict[gpu_id → SDKContext]
    │        (AscendC, __aicore__, DataCopy, 手动内存管理...)
    ▼
SpecAnalyzer
    │ 产出: OperatorIR (含 backward 字段)
    │        (name="silu", math="y = x * sigmoid(x)",
    │         backward_math="grad_x = grad_y * sigmoid(x) * ...",
    │         saved_for_backward=["x"],
    │         inputs/outputs...)
    ▼
TilingAgent
    │ 产出: TilingConfig
    │        (tile_length=21760, double_buffer=True, constraints...)
    ▼
CodeGenAgent.run()  [Forward 生成]
    │ 输入: OperatorIR + GPUSpec + SDKContext + TilingConfig
    │        + 错误知识库 + 禁用API列表(CUDA/HIP/AscendC)
    │        + Few-shot 示例(按 verification_level + BW 排序)
    │ [LLM 缓存: 命中 → 直接返回, 未命中 → 调 LLM → 存缓存]
    │ 产出: GeneratedKernel — forward (193 行 AscendC C++ 代码)
    ▼
CodeGenAgent.generate_backward()  [Backward 生成]
    │ 输入: OperatorIR.backward_math_description + forward_kernel
    │ [LLM 缓存: 同上]
    │ 产出: GeneratedKernel — backward (silu_backward AscendC C++)
    ▼
ReviewLoop ──┐
    │        │ 6 级硬件自适应验证 (VerifierAgent + HardwareDetector)
    │        │
    │        │ Level 1: 静态分析    → always
    │        │ Level 2: LLM 审查    → if llm_client
    │        │ Level 3: CPU 数学    → if torch (forward + backward gradcheck)
    │        │ Level 4: 真实编译    → if compiler (nvcc/hipcc/cann)
    │        │ Level 5: 硬件数值    → if GPU/NPU
    │        │ Level 6: Benchmark   → if GPU/NPU
    │        │
    │        │ checkpoint/resume: 每 stage 存 .review_checkpoints/
    │        │ progress callback: stage_name, iteration, passed → CLI
    │        │
    │        │ 失败时: → CodeGenAgent (重新生成)
    │        │         → OptimizerAgent (性能优化)
    │        │         → CompileErrorKB (记录错误 → 禁用列表更新)
    │        │
    │ 产出: ReviewSummary {final_kernel, passed=True, iterations=2,
    │        verification_level="benchmarked"}
    ▼
OperatorRegistry (SQLite)
    │ 存储: OperatorEntry {代码, 验证结果, 版本号, verification_level}
    ▼
Distribution (如果多 GPU)
    │ 产出: DistributionPlan {负载权重, 通信后端, 张量分片}
    ▼
TrainingExecutor (如果有训练代码)
    │ 步骤: 1. 加载算子  2. torch.library 注册
    │       3. 生成 torch.autograd.Function 包装 (forward+backward)
    │       4. 修改训练脚本  5. 生成启动命令
    │ 产出: TrainingJob {启动命令, 注入的算子, autograd wrappers}
    ▼
RuntimeMonitor (如果启动了训练)
    │ 产出: MonitorReport {GPU利用率, 显存, Loss, 告警}
    ▼
最终输出:
  ├── output/silu_ascend_910b.cpp       (forward 源代码文件)
  ├── output/silu_backward_ascend_910b.cpp  (backward 源代码文件)
  ├── .operator_registry.db              (算子仓库更新)
  ├── .llm_cache.db                      (LLM 缓存更新)
  └── 控制台: ✅ Generated: 193 chars, backend=ascendc
              📊 Review: passed=True, iters=2, BW=72%
```

---

## 关键设计总结

| 设计 | 为什么这样做 |
|------|------------|
| **双路径编排** | 已有算子直接复用，避免浪费 LLM 调用 |
| **6 级硬件自适应验证** | 自动检测硬件，验到能力范围的最深层；无 GPU 照样能出静态+CPU 验证结果 |
| **checkpoint/resume** | Review Loop 每阶段存盘，crash 后自动恢复，不浪费已完成的验证 |
| **progress callback** | CLI 实时显示验证进度，用户不再黑盒等待 |
| **Backward 代码生成** | 自动生成反向 kernel + gradcheck 验证，让生成的算子真正可用于训练 |
| **autograd.Function 包装** | TrainingExecutor 自动生成 PyTorch autograd 兼容层，无缝替换原始算子 |
| **LLM 缓存 (SQLite)** | 相同请求直接返回缓存，大幅节省 API 调用成本和时间 |
| **编译错误知识库** | 从失败中学习，越用越聪明 |
| **三后端禁用 API 列表** | CUDA/HIP/AscendC 各自维护禁用列表，有效减少 LLM 幻觉导致的编译失败 |
| **Few-shot 排序优化** | 按 verification_level + bandwidth_utilization 双维度选最佳历史代码 |
| **代码质量自动降级** | 宁可生成简单但能编译的代码，也不死循环 |
| **上下文累积** | 每次修复都带着完整历史，不重蹈覆辙 |
| **算子仓库** | 一次生成，永久复用；支持搜索、导出、版本历史 |
| **模板 + LLM** | 常见算子用模板（快速可靠），未知算子用 LLM（灵活） |
| **多 LLM 后端** | 不绑定单一供应商，可灵活切换 (Qwen/OpenAI/Anthropic) |
| **CLI 命令丰富** | `registry search`、`kb export/import`、`cache stats/clear` 一站式管理 |
