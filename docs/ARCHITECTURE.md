# 系统架构详解

> 本文档面向需要理解或修改系统内部结构的工程师，描述各子系统的职责、数据流和关键接口。
> 
> **配套示例文档**：[EXAMPLES.md](EXAMPLES.md) — 通过四个具体例子（含真实日志和代码）展示系统运作全过程。

---

## 目录

- [整体架构图](#整体架构图)
- [核心数据流](#核心数据流)
- [operators/ 子系统](#operators-子系统)
- [agents/ 子系统](#agents-子系统)
- [算子复杂度分类体系](#算子复杂度分类体系)
- [持久化存储层](#持久化存储层)
- [知识库层](#知识库层)

---

## 整体架构图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              用户接口层                                   │
│                                                                          │
│   examples/full_agent_lora_train.py     cli.py     main.py              │
│   (主流程 Step 0→5)                    (CLI)      (Python API)           │
└───────────────────────────┬──────────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────────┐
│                           agents/ 子系统                                  │
│                                                                          │
│   TrainingAnalystAgent   →  分析训练代码，输出 TrainingPlan               │
│   AutoOpRegistrar        →  识别缺失算子，生成 OperatorDesc               │
│   OperatorSpecAgent      →  算子规格解析，输出 OperatorIR                 │
│   CodeGenAgent           →  调用 LLM 生成 CUDA kernel                    │
│   (+ 硬件分析/验证/优化/训练执行等 10 个 Agent)                           │
└───────────────────────────┬──────────────────────────────────────────────┘
                            │
           ┌────────────────┼────────────────┐
           │                │                │
┌──────────▼──────┐ ┌───────▼──────┐ ┌──────▼──────────────────────┐
│  operators/ 子系统│ │  models/     │ │  knowledge_base/            │
│                  │ │              │ │                              │
│  op_desc.py      │ │  operator_ir │ │  gpu_database.py            │
│  op_registry.py  │ │  hardware    │ │  compile_error_kb.py        │
│  verify.py       │ │  _model      │ │                              │
│  patch.py        │ └──────────────┘ └──────────────────────────────┘
│  builtin_ops.py  │
│  auto_registrar.py│                  ┌──────────────────────────────┐
│  registry.py     │                  │  tools/                      │
│  generated_ops.py│ ←──────────────→ │  llm_client.py               │
└──────────────────┘                  │  cpu_simulator.py            │
                                      │  model_router.py             │
                                      └──────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────────┐
│                        外部依赖                                           │
│                                                                          │
│   Qwen API (DashScope)    nvcc (CUDA Toolkit)    SQLite (.db)            │
│   transformers / PEFT     PyTorch 2.x             GPU (CUDA)             │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 核心数据流

```
用户输入（训练脚本路径 + 模型名）
    │
    ▼
[Step 0] TrainingAnalystAgent
    输入: 训练脚本源码
    输出: TrainingPlan
           .required_operators = ["silu", "rmsnorm", "matmul", ...]
           .critical_operators = ["silu", "rmsnorm"]
           .model_architecture = "qwen"
    │
    ▼
[Step 0b] AutoOpRegistrar.find_missing(plan, op_registry)
    输入: TrainingPlan + OpRegistry（已注册的算子）
    输出: missing_ops = ["matmul"]  （silu/rmsnorm 已在 builtin_ops 中注册）
    │
    ▼
[Step 0b] AutoOpRegistrar.generate_missing_descs(missing_ops)
    输入: ["matmul"]
    输出: [OperatorDesc(name="matmul", ctypes=["void*","void*","void*","int","int","int"],...)]
    同时写入 operators/generated_ops.py
    │
    ▼
[Step 0c] 查询持久化 OperatorRegistry（SQLite）
    输入: kernel_names = ["silu_forward", "silu_backward", ...]
    输出: cached_kernels = {name: {"code": str, "flags": list}}（若已缓存）
    如果全部命中 → 跳转到 Step 2（重新编译 .so）
    │
    ▼（未命中缓存时）
[Step 1] CodeGenAgent（for each op in plan）
    输入: OperatorIR + GPUSpec
    输出: kernels = {
        "silu_forward":  {"code": "...", "flags": ["-O3", "-arch=sm_89"]},
        "silu_backward": {"code": "...", "flags": [...]},
        "rmsnorm_forward":  {...},
        "rmsnorm_backward": {...},
    }
    编译失败 retry: 最多 3 次，第 3 次降级为 float-only 简单版本
    │
    ▼
[Step 2] compile_all_kernels()
    输入: kernels（代码字符串）
    输出: so_paths = {
        "silu_forward":  "/tmp/kernels/silu_forward.so",  （成功）
        "rmsnorm_backward": None,                          （失败 → fallback）
    }
    │
    ▼
[Step 2.5] verify_all_kernels_generic(op_registry, so_paths)
    输入: OpRegistry（含 OperatorDesc）+ so_paths
    对每个 key：
        1. 按 OperatorDesc.ctypes_argtypes 加载 .so
        2. 按 input_shapes_fn 生成测试输入
        3. 调用 kernel → 与 pytorch_reference 对比
        4. 相对误差 < error_threshold（默认 0.05）→ 通过
    验证失败: so_paths[key] = None
    通过的 kernel 写入 SQLite（OperatorRegistry）
    输出: verified_so_paths + verify_report
    │
    ▼
[Step 3] patch_model(model, op_registry, fn_map)
    输入: Qwen3-8B + OpRegistry + fn_map（build_custom_fn_map 从 so_paths 构建）
    按各算子的 inject_pattern 遍历模型：
        silu:    替换 36 个 module.act_fn（type 含 "silu"）
        rmsnorm: 替换 145 个 Qwen3RMSNorm 模块
        matmul:  替换 252 个 nn.Linear（可选）
    输出: 注入后的 model（原地修改）
    │
    ▼
[Step 4] LoRA 训练（PEFT）
    输入: 注入后的模型 + Alpaca 数据集
    步数: 300 steps
    输出: 微调后的模型权重
    │
    ▼
[Step 5] 评估
    Custom / Baseline / No-finetune 三模式指令跟随率对比
```

---

## operators/ 子系统

### op_desc.py — OperatorDesc

算子的完整描述，系统所有自动化操作的"规格书"。

```
OperatorDesc
├── A. 身份信息
│   ├── name: str            ("silu", "rmsnorm", ...)
│   └── variant: str         ("forward" | "backward")
│       key = f"{name}_{variant}"
│
├── B. ctypes 接口规格（verify_kernel 使用）
│   ├── ctypes_argtypes: list[str]      类型字符串列表
│   │   支持: "void*", "float*", "half*", "int*",
│   │         "int", "int32", "int32_t", "int64", "int64_t",
│   │         "float", "double", "bool"
│   ├── output_arg_indices: list[int]   哪些参数位置是输出
│   └── output_dtypes: list[str]        每个输出的精度 (fp16/fp32/bf16)
│
├── C. PyTorch 参考实现（verify 数值对比用）
│   ├── pytorch_reference: Callable     fn(*inputs) -> Tensor | tuple
│   ├── input_shapes_fn: Callable       fn(test_case) -> dict[str, Tensor]
│   ├── scalar_args_fn: Callable        fn(test_case, inputs) -> list
│   ├── output_shapes_fn: Callable      自定义输出 shape（可选）
│   ├── test_cases: list[dict]          测试用例（默认 3 个标准尺寸）
│   └── error_threshold: float          相对误差阈值（默认 0.05）
│
└── D. 模型注入规格（patch_model 使用）
    ├── inject_pattern: tuple | Callable | None
    │   四种格式：
    │   ("attr", attr_name, type_substr)  → 替换属性
    │   ("module_type", type_substr)      → 替换整个子模块
    │   ("linear_name", name_substr)      → 替换 nn.Linear
    │   callable(name, module) -> bool    → 自定义匹配
    └── inject_fn: Callable | None
        fn(desc, so_paths) -> dict[str, object]
        负责加载 .so，创建 autograd.Function 包装，返回注入对象
```

关键方法：
- `desc.key` → `"silu_forward"`
- `desc.resolved_ctypes_argtypes()` → `[c_void_p, c_void_p, c_int]`
- `desc.resolved_output_dtypes()` → `["fp16"]`
- `desc.default_test_cases()` → 3 个标准测试用例

---

### op_registry.py — OpRegistry

运行时算子注册中心，以 `key` 为索引。

```python
from operators.op_registry import get_op_registry
reg = get_op_registry()   # 全局单例

# 注册
reg.register(my_desc)

# 查找
desc = reg.get("silu_forward")        # 精确 key
descs = reg.get_by_name("silu")       # 同名的所有 variant

# 枚举
names = reg.names()                   # 所有算子名（去重）
fwd_descs = reg.get_forward_descs()  # 所有 forward variant（patch_model 使用）

# 构建注入函数映射
fn_map = reg.build_custom_fn_map(so_paths)  # 调用各 desc.inject_fn
```

---

### verify.py — verify_kernel

从 `OperatorDesc` 自动构建 ctypes 调用并数值验证，无需手写任何验证代码。

```python
from operators.verify import verify_kernel, verify_all_kernels_generic

# 验证单个 kernel
result = verify_kernel(desc, so_path="/path/to/silu_forward.so")
# result = {"passed": True, "rel_err": 0.008, "detail": "...", "n_cases": 3}

# 批量验证（用于 Step 2.5）
verified_paths, verify_report = verify_all_kernels_generic(
    op_registry=reg,
    so_paths={"silu_forward": "/path/to/silu_forward.so", ...}
)
```

内部流程（每个 test_case）：
1. `input_shapes_fn(tc)` → 生成输入张量（自动 `.to(device).contiguous()`）
2. 为每个输出分配 buffer（float32 输出预置零，防 atomicAdd 累加问题）
3. `scalar_args_fn(tc, inputs)` → 生成标量参数
4. 按 `ctypes_argtypes` 顺序拼装调用参数
5. `fn(*call_args); torch.cuda.synchronize()`
6. NaN/Inf 检查
7. 与 `pytorch_reference` 对比相对误差（backward variant 不包在 no_grad 里）

---

### patch.py — patch_model

通用模型注入，支持 4 种注入模式。

```python
from operators.patch import patch_model

# fn_map 由 op_registry.build_custom_fn_map(so_paths) 生成
counts = patch_model(model, op_registry, fn_map)
# counts = {"silu": 36, "rmsnorm": 145, "matmul": 252}
```

四种注入模式详解：

| 模式 | 格式 | 适用场景 | 实现说明 |
|------|------|---------|---------|
| attr | `("attr", "act_fn", "silu")` | 替换激活函数属性 | `object.__setattr__` 绕过 nn.Module 类型检查 |
| module_type | `("module_type", "RMSNorm")` | 替换整个子模块 | 递归查找类型名含子串的模块，调用 `inject_obj(weight, eps)` 实例化 |
| linear_name | `("linear_name", "")` | 替换所有/部分 nn.Linear | name_substr="" 匹配全部 Linear |
| callable | `lambda name, mod: ...` | 自定义匹配逻辑 | 完全由用户控制匹配条件 |

---

### builtin_ops.py — 内置算子描述

定义了 10 个内置 `OperatorDesc` 并提供 `register_builtin_ops(registry)` 注册函数。

各算子 `inject_fn` 工厂的行为：
- `_make_silu_inject_fn`：加载 `silu_forward.so` 和 `silu_backward.so`，创建 `SiLUCustomFunction(autograd.Function)`，返回 `{"silu_fn": callable}`
- `_make_rmsnorm_inject_fn`：加载两个 .so，创建 `RMSNormCustomModule(nn.Module)`，返回 `{"RMSNormModule": class}`（需实例化时传 weight + eps）
- `_make_gelu_inject_fn`：同 SiLU 模式，返回 `{"gelu_fn": callable}`
- `_make_linear_inject_fn`：加载 `matmul_forward.so`，创建 `CustomLinear(nn.Module)`（包装原始 `nn.Linear` 的 weight/bias），返回 `{"LinearModule": class}`

所有 inject_fn 都实现了 **fallback 机制**：若 .so 加载失败，自动降级为 PyTorch 实现，不影响训练。

---

### auto_registrar.py — AutoOpRegistrar

```
AutoOpRegistrar
├── infer_strategy(op_name) → str
│   优先级: OPERATOR_TEMPLATES.category > 名称关键词匹配
│   结果: "elementwise" | "normalization" | "matmul" |
│         "reduction" | "embedding" | "complex" | "skip" | "unknown"
│
├── find_missing(plan, registry) → list[str]
│   对比 plan.all_operators() vs registry.names()
│   排除 "skip"（通信类）和 "unknown"
│   "complex" 仍加入列表，但 generate_op_desc 返回 None + 警告
│
├── generate_op_desc(op_name) → OperatorDesc | None
│   按 strategy 路由到对应模板构造器
│   complex/skip/unknown → 返回 None
│
├── generate_missing_descs(ops) → list[OperatorDesc]
│   批量生成，打印每个算子的复杂度说明
│
└── write_and_register(descs, registry, path)
    幂等写入 operators/generated_ops.py（跳过已存在的 key）
    同时注册到运行时 OpRegistry
```

---

### registry.py — 持久化 OperatorRegistry

SQLite 后端的算子仓库，存储已验证 kernel 的源码和元数据。

```python
from operators.registry import get_registry, OperatorEntry

reg = get_registry()  # 全局单例（线程安全，WAL 模式）

# 存入
reg.register(OperatorEntry(
    operator_name="silu_forward",
    gpu_model="NVIDIA GeForce RTX 4090",
    backend="cuda",
    source_code="...",
    build_flags=["-O3", "-arch=sm_89"],
    correctness_passed=True,
    max_relative_error=0.008,
    verification_level="hw_verified",
))

# 查找
entry = reg.lookup("silu_forward", "NVIDIA GeForce RTX 4090")
if entry and entry.correctness_passed:
    use(entry.source_code)
```

`OperatorEntry.verification_level` 枚举：
`none` → `static` → `llm_review` → `cpu_math` → `compiled` → `hw_verified` → `benchmarked`

---

## agents/ 子系统

### 核心 Agent 职责

| Agent | 文件 | 主要输入 | 主要输出 |
|-------|------|---------|---------|
| TrainingAnalystAgent | training_analyst.py | 训练脚本源码 | TrainingPlan |
| OperatorSpecAgent | spec_analyzer.py | 算子名字符串 | OperatorIR |
| CodeGenAgent | code_generator.py | OperatorIR + GPUSpec | GeneratedKernel（含源码） |
| HardwareProfilerAgent | hardware_profiler.py | GPU Key | GPUSpec |
| VerifierAgent | verifier.py | GeneratedKernel + GPUSpec | 验证报告 |
| ReviewLoopAgent | review_loop.py | 算子请求 | 经过 5 阶段质量循环的 kernel |
| OptimizerAgent | optimizer.py | GeneratedKernel + GPUSpec | 优化后的 kernel |
| TrainingExecutorAgent | training_executor.py | 注入后的模型 | 训练结果 |

### CodeGenAgent — 编译 Retry 机制

```
第 1 次: 正常 LLM 生成 → 编译
         失败 → 将 stderr 加入 fix_context
第 2 次: LLM 根据 fix_context 修复 → 编译
         失败 → 再次更新 fix_context
第 3 次: 降级 prompt（简单 float-only 版本）→ 编译
         失败 → so_path = None（PyTorch fallback）
```

`fix_context` 通过 `AgentContext.set_artifact("fix_context", error_text)` 传递给下一次生成。

### TrainingAnalystAgent — 算子识别

基于两种机制识别训练代码中使用的算子：

1. **OP_KEYWORD_MAP**：正则匹配（如 `r"\bsilu\b|SiLU|F\.silu"` → `"silu"`）
2. **ARCHITECTURE_OP_MAP**：按模型架构推断（如 `"qwen"` → `["flash_attention", "rmsnorm", "silu", "matmul", "embedding"]`）

---

## 算子复杂度分类体系

```
OperatorCategory（来自 models/operator_ir.py）
│
├── ELEMENTWISE   → strategy: "elementwise"
│   ctypes: (x: void*, out: void*, N: int)
│   适用: silu, gelu, relu, tanh, sigmoid, hardswish, mish, ...
│   注入: ("attr", "act_fn", <op_name>)
│
├── NORMALIZATION → strategy: "normalization"
│   ctypes（无 bias）: (x, w, out: void*, N, H: int, eps: float)
│   ctypes（含 bias）: (x, w, bias, out: void*, N, H: int, eps: float)
│   适用: rmsnorm, layernorm, batchnorm, groupnorm
│   注入: ("module_type", <TypeName>)
│
├── MATMUL        → strategy: "matmul"
│   ctypes: (A, B, C: void*, M, N, K: int)
│   适用: matmul, linear, gemm, bmm
│   注入: ("linear_name", "") 替换所有 nn.Linear
│
├── REDUCTION     → strategy: "reduction"
│   ctypes: (x, out: void*, N, C: int)
│   适用: softmax, log_softmax, cross_entropy
│   注入: 暂不注入（Qwen3 内部 attention 中调用，注入复杂度高）
│
├── EMBEDDING     → strategy: "embedding"
│   ctypes: (weight, indices, out: void*, V, H: int)
│   适用: embedding, embed_tokens
│   注入: 暂不注入
│
├── ATTENTION     → strategy: "complex"
│   接口依赖运行时参数（batch_size, num_heads, seq_len, head_dim）
│   ctypes 接口无法静态推导
│   → AutoOpRegistrar 返回 None，使用 PyTorch fallback
│
├── FUSED         → strategy: "complex"
│   融合算子（如 fused_moe），接口高度定制
│   → 暂不支持自动生成
│
└── COMMUNICATION → strategy: "skip"
    集合通信（allreduce/allgather）
    不适合 ctypes 单机注入
    → 直接跳过
```

---

## 持久化存储层

系统使用两个独立的 SQLite 数据库：

### .operator_registry.db（OperatorRegistry）
位置：项目根目录下。存储已验证 kernel 的完整源码、build_flags、验证结果。

主要表：
- `operators`：(operator_name, gpu_model, backend, source_code, build_flags, correctness_passed, max_relative_error, verification_level, version, ...)

特性：
- WAL 模式 + 线程本地连接（并发安全）
- 支持版本历史（同一 key 多次注册）
- `registry_key = "{operator_name}::{gpu_model}"`

### .operator_registry.json（旧版 JSON，自动迁移）
首次启动时自动检测并迁移到 SQLite。

---

## 知识库层

### knowledge_base/hardware_specs/gpu_database.py
10 款 GPU 的完整规格（memory_bw、flops、sm_count 等），供 CodeGenAgent 和 TilingAgent 使用。

```python
from knowledge_base.hardware_specs.gpu_database import get_gpu_spec
spec = get_gpu_spec("rtx_4090")
# spec.memory_bandwidth_gbps, spec.fp16_tflops, ...
```

### knowledge_base/compile_error_kb.py
37 条编译错误自动修复规则 + 动态学习。

当 nvcc 报错时：
1. 先查知识库，尝试自动 patch（如 `__shfl_down` → `__shfl_down_sync`）
2. 若无匹配规则，将错误加入 `fix_context` 反馈给 LLM
3. LLM 修复成功后，可将新规则写入知识库供后续复用
```

---
