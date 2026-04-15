# 算子子系统详解（operators/）

> 本文档详细介绍 `operators/` 目录下的通用算子框架：数据结构、接口约定、内置算子列表、如何添加新算子，以及 AutoOpRegistrar 的工作原理。

---

## 目录

- [设计目标](#设计目标)
- [OperatorDesc 字段完整说明](#operatordesc-字段完整说明)
- [ctypes 接口约定](#ctypes-接口约定)
- [内置算子列表](#内置算子列表)
- [添加新算子的完整步骤](#添加新算子的完整步骤)
- [AutoOpRegistrar 工作原理](#autoopregistrar-工作原理)
- [验证机制（verify_kernel）详解](#验证机制verify_kernel详解)
- [注入机制（patch_model）详解](#注入机制patch_model详解)

---

## 设计目标

`operators/` 子系统的设计目标是：**用一份 `OperatorDesc` 描述，驱动算子的生成、验证、注入三个阶段，不重复编写任何代码。**

```
OperatorDesc（一份描述）
    │
    ├──→ verify_kernel()   自动构建 ctypes 调用，对比 PyTorch reference
    ├──→ patch_model()     按 inject_pattern 找到注入点，替换模型算子
    └──→ AutoOpRegistrar   根据描述自动生成 CUDA kernel 的 LLM prompt
```

---

## OperatorDesc 字段完整说明

```python
@dataclass
class OperatorDesc:
    # ── A. 身份信息 ────────────────────────────────────────────────
    name: str               # 算子名，如 "silu"、"rmsnorm"
    variant: str = "forward"  # "forward" 或 "backward"
    # key = f"{name}_{variant}"，用于 OpRegistry 索引

    # ── B. ctypes 接口规格 ─────────────────────────────────────────
    ctypes_argtypes: list[str]
    # 对应 .so 中 void launch_kernel(...) 的参数类型列表
    # 类型字符串：void*/float*/half*/int*/int/int32/int32_t/int64/float/double/bool
    # GPU 指针（void*, float*, half*, int*）统一映射为 ctypes.c_void_p
    
    output_arg_indices: list[int]
    # ctypes_argtypes 中输出参数的位置索引（从 0 开始）
    # 这些位置对应预分配的输出 buffer

    output_dtypes: list[str]
    # 每个输出 buffer 的精度，与 output_arg_indices 一一对应
    # 可选值：fp16 / bf16 / fp32 / int32 / int64
    # backward kernel 的梯度通常用 fp32（防止 fp16 上溢→NaN）
    # 若为空，自动填充 fp16

    # ── C. PyTorch 参考实现 ────────────────────────────────────────
    pytorch_reference: Optional[Callable]
    # 签名：fn(*inputs) -> Tensor | tuple[Tensor, ...]
    # inputs 按 input_shapes_fn 返回字典的 value 顺序传入
    # 若为 None，验证时只做 NaN 检查

    input_shapes_fn: Optional[Callable]
    # 签名：fn(test_case: dict) -> dict[str, Tensor]
    # test_case 是 {"N": 64, "H": 1024} 形式的参数字典
    # 返回字典的 value 按序对应 ctypes_argtypes 中【非输出】的指针参数

    scalar_args_fn: Optional[Callable]
    # 签名：fn(test_case: dict, input_tensors: dict) -> list
    # 返回 ctypes_argtypes 中所有非指针参数（int/float）的值列表
    # 按 ctypes_argtypes 中非指针类型的出现顺序排列

    output_shapes_fn: Optional[Callable]
    # （可选）自定义输出 buffer shape
    # 签名：fn(test_case, input_tensors, out_idx: int) -> tuple[int, ...]
    # 若为 None，默认使用 (first_input.numel(),)（flat 一维）

    test_cases: list[dict]
    # 测试用例列表，每个 dict 包含 input_shapes_fn/scalar_args_fn 需要的参数
    # 若为空，使用 default_test_cases() = [{N:64,H:1024}, {N:8,H:3072}, {N:16,H:1024}]

    error_threshold: float = 0.05
    # 相对误差阈值，低于此值视为验证通过（默认 5%）

    # ── D. 模型注入规格 ────────────────────────────────────────────
    inject_pattern: InjectPattern
    # 模型中注入点的定位模式（详见下文）
    # None → 该算子不注入（backward / 仅验证的算子）

    inject_fn: Optional[Callable]
    # 工厂函数，签名：fn(desc, so_paths: dict) -> dict[str, object]
    # so_paths = {"silu_forward": "/path/to/silu_forward.so", ...}
    # 返回值并入 fn_map，供 patch_model 使用
    # None → 不注入
```

---

## ctypes 接口约定

系统要求所有 CUDA kernel 的 C 接口统一为：

```c
// 入口函数名必须为 launch_kernel（固定）
extern "C" void launch_kernel(/* 参数列表 */);
```

GPU 指针参数使用 `half*`（fp16）或 `float*`（fp32），在 Python 侧统一用 `ctypes.c_void_p` 传递 `.data_ptr()`。

### 各算子类别的标准 ctypes 接口

#### ELEMENTWISE 类（以 SiLU 为例）

```c
// void launch_kernel(half* x, half* out, int numel)
ctypes_argtypes = ["void*", "void*", "int"]
output_arg_indices = [1]   // out 在 index=1
output_dtypes = ["fp16"]
```

```c
// SiLU backward: void launch_kernel(half* go, half* x, float* grad_in, int numel)
ctypes_argtypes = ["void*", "void*", "void*", "int"]
output_arg_indices = [2]   // grad_in 在 index=2
output_dtypes = ["fp32"]   // 梯度用 fp32 防溢出
```

#### NORMALIZATION 类（RMSNorm forward）

```c
// void launch_kernel(half* x, half* weight, half* out, int N, int H, float eps)
ctypes_argtypes = ["void*", "void*", "void*", "int", "int", "float"]
output_arg_indices = [2]   // out 在 index=2
output_dtypes = ["fp16"]
```

```c
// RMSNorm backward:
// void launch_kernel(half* go, half* x, half* weight,
//                    float* grad_x, float* grad_w,
//                    int N, int H, float eps)
ctypes_argtypes = ["void*","void*","void*","void*","void*","int","int","float"]
output_arg_indices = [3, 4]    // 两个梯度输出
output_dtypes = ["fp32", "fp32"]
```

#### MATMUL 类

```c
// void launch_kernel(half* A, half* B, half* C, int M, int N, int K)
// A: (M×K), B: (K×N), C: (M×N)
ctypes_argtypes = ["void*", "void*", "void*", "int", "int", "int"]
output_arg_indices = [2]
output_dtypes = ["fp16"]
```

#### REDUCTION 类（Softmax）

```c
// void launch_kernel(half* x, half* out, int N, int C)
// 对每行 C 个元素做 softmax
ctypes_argtypes = ["void*", "void*", "int", "int"]
output_arg_indices = [1]
output_dtypes = ["fp16"]
```

#### EMBEDDING 类

```c
// void launch_kernel(half* weight, int* indices, half* out, int vocab_size, int hidden)
ctypes_argtypes = ["void*", "void*", "void*", "int", "int"]
output_arg_indices = [2]
output_dtypes = ["fp16"]
```

---

## 内置算子列表

以下算子在 `operators/builtin_ops.py` 中有完整定义，开箱即用：

### SiLU（逐元素激活）

```python
# forward
SILU_FORWARD_DESC = OperatorDesc(
    name="silu", variant="forward",
    ctypes_argtypes=["void*", "void*", "int"],
    output_arg_indices=[1],
    output_dtypes=["fp16"],
    pytorch_reference=lambda x: F.silu(x),
    input_shapes_fn=lambda tc: {"x": torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda")},
    scalar_args_fn=lambda tc, inp: [inp["x"].numel()],
    inject_pattern=("attr", "act_fn", "silu"),   # 替换模型中 act_fn 属性
    inject_fn=_make_silu_inject_fn,
)

# backward（不注入，由 forward inject_fn 内部的 autograd.Function 处理）
SILU_BACKWARD_DESC = OperatorDesc(
    name="silu", variant="backward",
    ctypes_argtypes=["void*", "void*", "void*", "int"],  # go, x, grad_fp32, N
    output_arg_indices=[2],
    output_dtypes=["fp32"],
    inject_pattern=None,
)
```

### RMSNorm（归一化）

```python
RMSNORM_FORWARD_DESC = OperatorDesc(
    name="rmsnorm", variant="forward",
    ctypes_argtypes=["void*", "void*", "void*", "int", "int", "float"],
    output_arg_indices=[2],
    output_dtypes=["fp16"],
    pytorch_reference=lambda x, w: (x.float() / torch.sqrt(x.float().pow(2).mean(-1,keepdim=True)+1e-6)*w.float()).half(),
    input_shapes_fn=lambda tc: {
        "x":      torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda"),
        "weight": torch.ones(tc["H"],           dtype=torch.float16, device="cuda"),
    },
    scalar_args_fn=lambda tc, inp: [tc["N"], tc["H"], 1e-6],
    output_shapes_fn=lambda tc, inp, i: (tc["N"], tc["H"]),
    inject_pattern=("module_type", "RMSNorm"),   # 替换类型名含 "RMSNorm" 的模块
    inject_fn=_make_rmsnorm_inject_fn,
)
```

### GeLU（逐元素激活）

与 SiLU 结构相同，`inject_pattern=("attr", "act_fn", "gelu")`。

### MatMul/Linear（矩阵乘法）

```python
MATMUL_FORWARD_DESC = OperatorDesc(
    name="matmul", variant="forward",
    ctypes_argtypes=["void*", "void*", "void*", "int", "int", "int"],
    output_arg_indices=[2],
    output_dtypes=["fp16"],
    pytorch_reference=lambda A, B: torch.matmul(A.float(), B.float()).half(),
    input_shapes_fn=lambda tc: {
        "A": torch.randn(tc.get("M",64), tc.get("K",128), dtype=torch.float16, device="cuda"),
        "B": torch.randn(tc.get("K",128), tc.get("N",64), dtype=torch.float16, device="cuda"),
    },
    scalar_args_fn=lambda tc, inp: [tc.get("M",64), tc.get("N",64), tc.get("K",128)],
    test_cases=[{"M":64,"K":4096,"N":4096}, {"M":32,"K":4096,"N":12288}],
    inject_pattern=("linear_name", ""),   # "" 匹配所有 nn.Linear
    inject_fn=_make_linear_inject_fn,
)
```

`_make_linear_inject_fn` 创建的 `CustomLinear` 类：
- 保留原 `nn.Linear` 的 `weight`/`bias` 参数（Parameter）
- forward 用 GEMM kernel（2D 展平后调用），backward 用 PyTorch autograd 自动求导
- 失败时自动 fallback 到 `F.linear()`

### Softmax / CrossEntropy / Embedding

这三个算子只做生成和验证，`inject_pattern=None`，不注入模型：

| 算子 | ctypes 接口 | 备注 |
|------|------------|------|
| softmax | `(x*, out*, N, C)` | Qwen3 attention 内部调用，注入复杂度高，暂不替换 |
| cross_entropy | `(logits*, targets*, loss*, N, C)` | targets 是 int32，通过 void* 传递 |
| embedding | `(weight*, idx*, out*, V, H)` | idx 是 int32 |

---

## 添加新算子的完整步骤

### 场景 1：添加标准逐元素激活（如 Mish）

**第一步**：确认 Mish 属于 ELEMENTWISE 类型（ctypes 接口固定为 `(x*, out*, N)`）。

**第二步**：在 `operators/builtin_ops.py` 中添加描述：

```python
MISH_FORWARD_DESC = OperatorDesc(
    name="mish",
    variant="forward",
    ctypes_argtypes=["void*", "void*", "int"],
    output_arg_indices=[1],
    output_dtypes=["fp16"],
    pytorch_reference=lambda x: F.mish(x),
    input_shapes_fn=lambda tc: {
        "x": torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda")
    },
    scalar_args_fn=lambda tc, inp: [inp["x"].numel()],
    error_threshold=0.05,
    # 模型中的激活函数属性通常叫 act_fn，类型名含 "mish"
    inject_pattern=("attr", "act_fn", "mish"),
    inject_fn=_make_mish_inject_fn,   # 见下
)
```

**第三步**：实现 `inject_fn`：

```python
def _make_mish_inject_fn(desc: OperatorDesc, so_paths: dict) -> dict:
    forward_so = so_paths.get("mish_forward")
    forward_fn = None
    if forward_so and os.path.exists(forward_so):
        try:
            lib = ctypes.CDLL(forward_so)
            fn = lib.launch_kernel
            fn.restype = None
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            forward_fn = fn
        except Exception as e:
            logger.warning(f"Mish load failed: {e}")

    class MishCustomFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            x_c = x.contiguous()
            if forward_fn is not None:
                out = torch.empty_like(x_c)
                forward_fn(x_c.data_ptr(), out.data_ptr(), x_c.numel())
                torch.cuda.synchronize()
            else:
                out = F.mish(x)  # fallback
            ctx.save_for_backward(x)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            # 使用 PyTorch autograd 计算梯度
            xr = x.float().requires_grad_(True)
            F.mish(xr).backward(grad_output.float())
            return xr.grad.to(x.dtype)

    return {"mish_fn": lambda x: MishCustomFunction.apply(x)}
```

**第四步**：注册到 `ALL_BUILTIN_DESCS`：

```python
ALL_BUILTIN_DESCS.append(MISH_FORWARD_DESC)
```

**第五步**（可选）：在 `OPERATOR_TEMPLATES` 中添加 Mish 的规格模板，以便 `CodeGenAgent` 生成更精确的 prompt：

```python
# agents/spec_analyzer.py
OPERATOR_TEMPLATES["mish"] = {
    "category": OperatorCategory.ELEMENTWISE,
    "description": "Mish activation: x * tanh(softplus(x))",
    "math_description": "y = x * tanh(ln(1 + e^x))",
    # ...
}
```

---

### 场景 2：添加归一化算子（如 LayerNorm，含 bias）

LayerNorm 有三个输入（x, weight, bias），ctypes 接口：
`(x*, weight*, bias*, out*, N, H, eps)`

```python
LAYERNORM_FORWARD_DESC = OperatorDesc(
    name="layernorm",
    variant="forward",
    ctypes_argtypes=["void*", "void*", "void*", "void*", "int", "int", "float"],
    output_arg_indices=[3],           # out 在 index=3
    output_dtypes=["fp16"],
    pytorch_reference=lambda x, w, b: F.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float()).half(),
    input_shapes_fn=lambda tc: {
        "x":      torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda"),
        "weight": torch.ones(tc["H"],           dtype=torch.float16, device="cuda"),
        "bias":   torch.zeros(tc["H"],          dtype=torch.float16, device="cuda"),
    },
    scalar_args_fn=lambda tc, inp: [tc["N"], tc["H"], 1e-5],
    output_shapes_fn=lambda tc, inp, i: (tc["N"], tc["H"]),
    inject_pattern=("module_type", "LayerNorm"),
    inject_fn=_make_layernorm_inject_fn,
)
```

---

### 场景 3：添加不需要注入的验证算子

若只需生成+验证，不注入模型（如 `log_softmax`）：

```python
LOG_SOFTMAX_DESC = OperatorDesc(
    name="log_softmax",
    variant="forward",
    ctypes_argtypes=["void*", "void*", "int", "int"],
    output_arg_indices=[1],
    output_dtypes=["fp16"],
    pytorch_reference=lambda x, N, C: F.log_softmax(x.view(N,C).float(), dim=-1).half().view(N,C),
    input_shapes_fn=lambda tc: {"x": torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda")},
    scalar_args_fn=lambda tc, inp: [tc["N"], tc["H"]],
    inject_pattern=None,   # 不注入
    inject_fn=None,
)
```

---

### 使用新算子的完整流程

```python
from operators.op_registry import get_op_registry
from operators.builtin_ops import register_builtin_ops
from operators.verify import verify_kernel

# 1. 注册内置算子（含你新加的 Mish）
reg = register_builtin_ops(get_op_registry())

# 2. 验证
desc = reg.get("mish_forward")
result = verify_kernel(desc, so_path="/path/to/mish_forward.so")
print(result)  # {"passed": True, "rel_err": 0.003, ...}

# 3. 注入模型
from operators.patch import patch_model
so_paths = {"mish_forward": "/path/to/mish_forward.so"}
fn_map = reg.build_custom_fn_map(so_paths)
counts = patch_model(model, reg, fn_map)
print(counts)  # {"mish": 36}
```

---

## AutoOpRegistrar 工作原理

`AutoOpRegistrar` 的核心逻辑是：**通过 OperatorCategory（而不是硬编码的算子名列表）自动判断算子的 ctypes 接口模板**。

### 工作流程

```
训练计划（TrainingPlan）
    │
    ▼ find_missing(plan, registry)
    对比 plan.all_operators() vs registry.names()
    找到未注册的算子：missing = ["tanh", "leaky_relu", ...]
    │
    ▼ infer_strategy(op_name) for each op
    ┌─────────────────────────────────────┐
    │ 1. 查 OPERATOR_TEMPLATES           │
    │    若在其中：读取 OperatorCategory   │
    │    用 _CATEGORY_STRATEGY 映射       │
    │    → strategy                       │
    │                                     │
    │ 2. 若不在：名称关键词匹配            │
    │    "tanh" in _NAME_TO_CATEGORY_HINT │
    │    → "elementwise"                  │
    │                                     │
    │ 3. 完全无法识别 → "unknown"         │
    └─────────────────────────────────────┘
    │
    ▼ generate_op_desc(op_name, strategy)
    按 strategy 路由到对应构造器：
      elementwise   → _make_elementwise_desc()
      normalization → _make_normalization_desc()  （含/不含 bias 自动判断）
      matmul        → _make_matmul_desc()
      reduction     → _make_reduction_desc()
      embedding     → _make_embedding_desc()
      complex/skip  → None（警告，PyTorch fallback）
    │
    ▼ write_and_register(descs, registry, path)
    幂等写入 operators/generated_ops.py
    同时注册到 OpRegistry（运行时使用）
```

### 关键设计：OperatorCategory 优先于关键词匹配

当算子已在 `OPERATOR_TEMPLATES` 中定义时（如 `rmsnorm`、`flash_attention`），AutoOpRegistrar 优先读取其 `OperatorCategory`，确保与 `OperatorSpecAgent` 的理解一致。

只有当算子不在模板库中时，才回退到名称关键词匹配（`_NAME_TO_CATEGORY_HINT`）。

### 示例输出（operators/generated_ops.py 片段）

```python
# Auto-generated 2025-01-15 10:30 ──────────────────────────────────────

# strategy: elementwise
TANH_FORWARD_DESC = OperatorDesc(
    name='tanh',
    variant='forward',
    ctypes_argtypes=['void*', 'void*', 'int'],
    output_arg_indices=[1],
    output_dtypes=['fp16'],
    pytorch_reference=lambda x: torch.tanh(x.float()).half(),
    input_shapes_fn=lambda tc: {"x": torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda")},
    scalar_args_fn=lambda tc, inp: [inp["x"].numel()],
    test_cases=[],
    error_threshold=0.05,
    inject_pattern=None,
    inject_fn=None,   # 如需注入模型，请手动实现此工厂函数
)

def register_generated_ops_20250115_1030(registry):
    """注册本批次自动生成的算子（2025-01-15 10:30）"""
    registry.register(TANH_FORWARD_DESC)
```

> `generated_ops.py` 由系统自动写入，但可以手动编辑（如补充 `inject_fn`），系统下次运行时自动加载。

---

## 验证机制（verify_kernel）详解

### 调用参数拼装逻辑

`verify_kernel` 根据 `ctypes_argtypes` 自动分配输入/输出指针：

```
ctypes_argtypes = ["void*", "void*", "void*", "int", "int", "float"]
                    ↑                                ↑
               指针类型（GPU 指针）            非指针类型（标量）

output_arg_indices = [2]   → index=2 是输出 buffer

分配规则：
  index=0 (void*，非输出) → 从 input_tensors 取第 1 个 .data_ptr()
  index=1 (void*，非输出) → 从 input_tensors 取第 2 个 .data_ptr()
  index=2 (void*，是输出) → 新分配的 output_buffers[2].data_ptr()
  index=3 (int，非指针)   → 从 scalar_values 取第 1 个
  index=4 (int，非指针)   → 从 scalar_values 取第 2 个
  index=5 (float，非指针) → 从 scalar_values 取第 3 个
```

### backward variant 的特殊处理

backward kernel 的参考实现内部需要调用 `.backward()`，**不能包在 `torch.no_grad()` 里**：

```python
# verify.py 中的处理
if desc.variant == "backward":
    ref_outputs = desc.pytorch_reference(*ref_inputs)   # 不用 no_grad
else:
    with torch.no_grad():
        ref_outputs = desc.pytorch_reference(*ref_inputs)
```

### 相对误差计算

对接近零的元素，用绝对误差代替相对误差，避免除零问题：

```python
threshold = denom.mean() * 0.05 + 1e-4
mask = denom > threshold
if mask.any():
    rel_err = (abs_err[mask] / (denom[mask] + 1e-6)).max().item()
else:
    rel_err = abs_err.max().item()   # fallback 到绝对误差
```

---

## 注入机制（patch_model）详解

### inject_fn 工厂函数的职责

`inject_fn(desc, so_paths) -> dict[str, object]` 负责：
1. 从 `so_paths` 中获取相关 kernel 的路径
2. 用 `ctypes.CDLL` 加载 .so，绑定 `launch_kernel` 函数签名
3. 创建包装了 CUDA kernel 的 `torch.autograd.Function` 或 `nn.Module`
4. 返回字典，key 是 `fn_map` 中的查找键

### fn_map 的键名约定

`patch_model` 从 `fn_map` 中查找注入对象，查找键优先级（以 silu 为例）：
1. `"silu_fn"` ← inject_fn 返回的键
2. `"SiluModule"` ← 类名约定
3. `"SILUMODULE"` ← 大写
4. `"silu"` ← 算子名本身
5. 特殊兼容：`"RMSNormModule"`（rmsnorm），`"LinearModule"`（matmul/linear）

### 全量 Linear 注入的注意事项

`inject_pattern=("linear_name", "")` 会替换模型中**所有** `nn.Linear`（252 个）：

- 权重参数复用原始 `Parameter`，不拷贝（节省显存）
- backward 通过 PyTorch autograd 自动求导（不用 GEMM kernel）
- 显存消耗与原始模型相同
- 若 GEMM kernel 验证失败（so_path=None），自动降级为 `F.linear()`

如果只想替换部分 Linear，可以通过 `name_substr` 过滤：

```python
inject_pattern=("linear_name", "q_proj")   # 只替换 q_proj 层
inject_pattern=("linear_name", "gate")     # 只替换 gate_proj 层
```
```

---
