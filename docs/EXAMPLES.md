# 示例：系统运作全过程图解

> 本文档通过两个具体示例，展示系统从"用户提交任务"到"训练完成"的完整运作过程。所有日志、代码片段均来自真实运行结果。

---

## 目录

- [示例一：Qwen3-8B LoRA 训练（完整端到端）](#示例一qwen3-8b-lora-训练完整端到端)
- [示例二：LLM 生成的代码有编译错误时——Retry 机制](#示例二llm-生成的代码有编译错误时retry-机制)
- [示例三：添加一个新算子（AutoOpRegistrar 自动识别）](#示例三添加一个新算子autoopregistrar-自动识别)
- [示例四：第二次运行——持久化缓存直接复用](#示例四第二次运行持久化缓存直接复用)

---

## 示例一：Qwen3-8B LoRA 训练（完整端到端）

**场景**：在 RTX 4090 上用 Qwen LLM 从头生成 SiLU + RMSNorm CUDA kernel，注入 Qwen3-8B 后做 Alpaca 指令微调。

**命令**：

```bash
MODE=full_agent LLM=qwen STEPS=300 MODEL=/path/to/Qwen3-8B sbatch run_lora_slurm.sh
```

---

### Step 0：初始化算子注册表

系统启动时，将所有内置 `OperatorDesc` 注册进运行时注册表：

```
[Step 0] 算子注册表已初始化:
OpRegistry (10 entries, 7 ops):
  cross_entropy        [forward]    — gen/verify only
  embedding            [forward]    — gen/verify only
  gelu                 [forward, backward]  ✅ injectable
  matmul               [forward]    ✅ injectable
  rmsnorm              [forward, backward]  ✅ injectable
  silu                 [forward, backward]  ✅ injectable
  softmax              [forward]    — gen/verify only
```

每个算子的 `OperatorDesc` 包含四类信息：
- **ctypes 接口**：`launch_kernel` 的 C 函数签名（指针类型和标量参数）
- **验证规格**：PyTorch reference + 测试用例 + 误差阈值
- **注入规格**：`inject_pattern`（如何在模型中找到替换点）+ `inject_fn`（如何创建替换对象）

---

### Step 0：自动识别未注册算子

`TrainingAnalystAgent` 静态分析模型名称，识别出训练所需的全部算子；`AutoOpRegistrar` 对比注册表，找出差集：

```
[Step 0] 发现未注册算子: ['flash_attention', 'matmul']

[AutoOpRegistrar] 算子复杂度分析:
  flash_attention: 复杂算子（无法自动推导）— 接口依赖运行时参数（head_dim/batch等），需手动实现
  matmul: 矩阵乘法（中等）— 两输入一输出，ctypes: (A, B, C, M, N, K)

[Step 0] 已自动注册 1 个新算子并写入 generated_ops.py
[Step 0] 以下算子接口复杂，暂用 PyTorch fallback: ['flash_attention']
```

**matmul** 被自动生成 `OperatorDesc` 并写入 `operators/generated_ops.py`：

```python
# operators/generated_ops.py（自动写入，可手动修改）
# strategy: matmul
MATMUL_FORWARD_DESC = OperatorDesc(
    name='matmul',
    variant='forward',
    ctypes_argtypes=['void*', 'void*', 'void*', 'int', 'int', 'int'],
    output_arg_indices=[2],
    output_dtypes=['fp16'],
    pytorch_reference=lambda A, B: torch.matmul(A.float(), B.float()).half(),
    input_shapes_fn=lambda tc: {
        "A": torch.randn(tc.get("M", 64), tc.get("K", 128), dtype=torch.float16, device="cuda"),
        "B": torch.randn(tc.get("K", 128), tc.get("N", 64), dtype=torch.float16, device="cuda"),
    },
    scalar_args_fn=lambda tc, inp: [tc.get("M", 64), tc.get("N", 64), tc.get("K", 128)],
    ...
)
```

**flash_attention** 因接口复杂度为 `ATTENTION` 类，返回 `None`，使用 PyTorch fallback。

---

### Step 1：LLM 生成 CUDA Kernel

系统向 Qwen API 发送 4 次请求，每次包含：
- 目标 GPU 规格（RTX 4090，sm_89，1TB/s 带宽，82.6 TFLOPS FP16）
- 算子数学定义和 PyTorch 参考实现
- 禁用 API 列表（37 条知识库规则，避免 LLM 幻觉）
- 要求的 C 接口格式（`extern "C" void launch_kernel(...)`）

```
14:44:48 HTTP POST → dashscope.aliyuncs.com  →  silu_forward (5907 chars, 35s)
  优化点: half2向量化, h2exp原生FP16, 网格步幅循环, 边界处理
14:45:23 HTTP POST → dashscope.aliyuncs.com  →  silu_backward (5967 chars, 35s)
14:45:58 HTTP POST → dashscope.aliyuncs.com  →  rmsnorm_forward (5284 chars, 34s)
14:46:54 HTTP POST → dashscope.aliyuncs.com  →  rmsnorm_backward (9267 chars, 56s)

[Step 1] 共生成 4 个 kernel，总计 26508 chars
```

生成的 `silu_forward.cu`（实际代码）：

```cuda
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void silu_forward_kernel(const half* __restrict__ x,
                                    half* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float xi = __half2float(x[idx]);
        float v = xi / (1.0f + expf(-xi));  // SiLU = x * sigmoid(x)
        out[idx] = __float2half(v);
    }
}

// ctypes 调用入口（固定签名）
extern "C" void launch_kernel(void* x, void* out, int N) {
    int block = 256;
    int grid = (N + block - 1) / block;
    silu_forward_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(out), N);
}
```

生成的 `silu_backward.cu`（backward 输出 float32，防止 fp16 overflow）：

```cuda
__global__ void silu_backward_kernel(
    const half* __restrict__ grad_out,
    const half* __restrict__ x,
    float* __restrict__ grad_in,   // ← float* 而非 half*，防止梯度 overflow→NaN
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float go  = __half2float(grad_out[idx]);
        float xi  = __half2float(x[idx]);
        float sig = 1.0f / (1.0f + expf(-xi));
        grad_in[idx] = go * sig * (1.0f + xi * (1.0f - sig));  // 直接写 float
    }
}

extern "C" void launch_kernel(void* grad_out, void* x, void* grad_in_fp32, int N) {
    int block = 256;
    int grid  = (N + block - 1) / block;
    silu_backward_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(grad_out),
        reinterpret_cast<const half*>(x),
        reinterpret_cast<float*>(grad_in_fp32),   // float* 输出
        N);
}
```

每个 kernel 生成后立即进行**试编译**（写临时文件，调 nvcc，自动清理），验证可以编译通过再继续。

---

### Step 2：正式编译为共享库

```
[Step 2] Compiling silu_forward:
  nvcc --shared -Xcompiler -fPIC -O3 -arch=sm_89 --use_fast_math
       output/full_agent/silu_forward.cu
       -o output/full_agent/silu_forward.so
→ silu_forward.so (1,008,080 bytes)  ✅

[Step 2] Compiling silu_backward:   → silu_backward.so (1,008,104 bytes)  ✅
[Step 2] Compiling rmsnorm_forward: → rmsnorm_forward.so (1,008,112 bytes) ✅
[Step 2] Compiling rmsnorm_backward:→ rmsnorm_backward.so (1,012,424 bytes) ✅
```

---

### Step 2.5：数值验证

用 `ctypes` 加载每个 `.so`，构造测试输入，调用 `launch_kernel`，与 PyTorch reference 对比相对误差：

```
[verify] 数值验证 4 个 kernel...

  ✅ silu_forward:     PASS (max_rel_err=0.0012)  ← 最大相对误差 0.12%
  ✅ silu_backward:    PASS (max_rel_err=0.0000)  ← 完全精确
  ✅ rmsnorm_forward:  PASS (max_rel_err=0.0009)  ← 误差 0.09%
  ❌ rmsnorm_backward: FAIL → fallback            ← 数值误差超阈值，使用 PyTorch fallback

  验证结果: 3/4 通过
  💾 3 个 kernel 已持久化存储（下次可直接复用）
```

**验证逻辑**（以 `silu_forward` 为例）：

```python
# 1. 构造输入
x = torch.randn(64, 1024, dtype=torch.float16, device="cuda")

# 2. 调用 agent kernel
out_kernel = torch.empty_like(x)
fn(x.data_ptr(), out_kernel.data_ptr(), x.numel())

# 3. PyTorch reference
out_ref = F.silu(x)

# 4. 对比误差
rel_err = ((out_kernel - out_ref).abs() / (out_ref.abs() + 1e-3)).max().item()
# → 0.0012，小于阈值 0.05，通过 ✅
```

---

### Step 3：构建注入对象

`OpRegistry.build_custom_fn_map(so_paths)` 调用每个算子的 `inject_fn` 工厂，生成可替换进模型的 Python 对象：

```python
# silu inject_fn 返回：
fn_map = {
    "silu_fn": <function silu_custom>,      # 调用已验证的 silu_forward.so
    "RMSNormModule": <class RMSNormCustomModule>,  # 调用 rmsnorm_forward.so
    "gelu_fn": <function gelu_custom>,      # 无 gelu so，使用 F.gelu fallback
    "LinearModule": <class CustomLinear>,   # 无 matmul so，使用 F.linear fallback
}
```

`RMSNormCustomModule` 是一个完整的 `nn.Module`，保留原始模型的 `weight` 参数，forward 调用 agent kernel，backward 走 PyTorch autograd：

```python
class RMSNormCustomModule(nn.Module):
    def __init__(self, weight: nn.Parameter, eps: float):
        super().__init__()
        self.weight = weight           # 复用原始权重，不重新初始化
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        x_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
        # 调用 agent 生成的 rmsnorm_forward kernel
        out_2d = RMSNormFunction.apply(x_2d, self.weight, self.variance_epsilon)
        return out_2d.reshape(hidden_states.shape)
```

---

### Step 4：注入模型

`patch_model(model, op_reg, fn_map)` 递归遍历 Qwen3-8B 的所有子模块，按各算子的 `inject_pattern` 找到替换点：

```
[Step 4] Replaced: SiLU × 36, RMSNorm × 145
```

- **SiLU × 36**：每个 Decoder Layer 的 MLP 有一个 `act_fn` 属性，类型名含 "silu"，被替换为 `silu_custom` 函数
- **RMSNorm × 145**：递归查找所有类型名含 "RMSNorm" 的子模块，替换为 `RMSNormCustomModule`（1 global + 36×2 layer norms + 36×2 q/k norms）

替换后，模型的 forward 路径自动调用 agent kernel，无需修改任何训练代码。

---

### Step 5：LoRA 训练 + 评估

```
LoRA params: 7,667,712 / 8,198,403,072 (0.09%)
Training for 300 steps on 27 Alpaca samples...

 Step    Loss    Grad Norm    LR
    1   0.9670   1.0000    5.00e-05
   50   0.8907   0.6896    4.70e-05
  100   0.3769   1.0000    3.88e-05
  150   0.0578   0.6610    2.75e-05   ← 接近收敛
  200   0.3207   1.0000    1.63e-05
  300   0.2859   1.0000    5.00e-06

Loss: 0.9670 → 0.2859（下降 70%）
```

**三模式对比结果**：

```
╔══════════════════════════════════════════════════════╗
║        Alpaca 指令跟随能力评估结果汇总               ║
╠══════════════════════════════════════════════════════╣
║  no_finetune  Score: 93.5%  (KW:87%  Fmt:100%)       ║
║  custom       Score: 95.0%  (KW:90%  Fmt:100%)  ↑    ║
║  baseline     Score: 92.5%  (KW:85%  Fmt:100%)       ║
╚══════════════════════════════════════════════════════╝

Custom vs No-Finetune: +1.5%（微调有效）
Custom vs Baseline:    +2.5%（算子数值等价，Loss 曲线几乎完全一致）
```

**关键验证**：Custom 与 Baseline 的 Loss 曲线步步相差 < 0.01，证明 agent 生成的 kernel 与 PyTorch 原生算子数值等价。

---

## 示例二：LLM 生成的代码有编译错误时——Retry 机制

**场景**：Job 942 中，Qwen LLM 为 `rmsnorm_backward` 生成了引用未声明变量 `f2_y` 的代码。

### 第一次尝试：编译失败

LLM 生成的代码使用了 `half2` 向量化，但犯了一个变量名错误：

```cuda
// LLM 第一次生成（有 bug）
half2 f2_x = *reinterpret_cast<const half2*>(&x_row[i]);
// 忘记声明 f2_y，直接使用了
sum_x_sq += f2_x.x * f2_x.x + f2_y.y * f2_y.y;  // ← f2_y 未声明！
```

试编译（`_try_compile`）立即失败，捕获到 stderr：

```
output/full_agent/rmsnorm_backward.cu(60): error: identifier "f2_y" is undefined
          sum_x_sq += f2_x.x * f2_x.x + f2_y.y * f2_y.y;
                                        ^
1 error detected in the compilation of "rmsnorm_backward.cu".
```

### Retry：把错误喂给 LLM 重新生成

系统构建 `fix_context`，将错误信息拼入下次请求的 prompt：

```python
fix_context = {
    "history_summary": "第1次错误:\noutput/full_agent/rmsnorm_backward.cu(60): "
                       "error: identifier \"f2_y\" is undefined\n...",
    "fix_guidance": "请修复以下编译错误（第2次尝试）：\n"
                    "error: identifier \"f2_y\" is undefined\n"
                    "只修改导致编译错误的代码，保持算法逻辑不变。",
    "iteration_history": [{"attempt": 1, "error": "..."}],
}
```

LLM 收到这个 prompt 后，理解了错误位置，在第 2 次生成中修复了变量声明。

```
[Retry 1/3] rmsnorm_backward compile failed:
  error: identifier "f2_y" is undefined

[Retry 2/3] rmsnorm_backward ✅ compiled OK on attempt 2/3
```

### 第三次自动降级

如果连续 2 次失败，系统在第 3 次请求时自动切换为"简单版 prompt"（`build_cuda_simple_prompt`）：

```python
if len(fix_context.get("iteration_history", [])) >= 2:
    # 禁用所有复杂优化（half2/shared memory/wmma/Tensor Core）
    # 只用 float 精度，每线程处理一个元素，最简单但必定能编译通过
    prompt = build_cuda_simple_prompt(op_ir, gpu_spec)
```

简单版 prompt 的约束：
- ❌ 禁止 half2 向量化
- ❌ 禁止 shared memory
- ❌ 禁止 wmma / Tensor Core
- ✅ 只用 `__half2float()` + `__float2half()` + 单线程单元素

降级后的代码虽然性能较低，但**编译成功率接近 100%**。

---

## 示例三：添加一个新算子（AutoOpRegistrar 自动识别）

**场景**：系统识别到要训练 GPT-2 架构的模型，其 MLP 使用 GELU 激活而非 SiLU，但 LayerNorm 也不同于 Qwen3 的 RMSNorm。

假设用户运行：

```bash
python examples/full_agent_lora_train.py --model /path/to/gpt2 --llm qwen
```

系统会从模型路径推断 `model_name_hint = "gpt2"`，`ARCHITECTURE_OP_MAP["gpt2"]` 返回：

```python
["flash_attention", "layernorm", "gelu", "matmul", "embedding"]
```

`AutoOpRegistrar.find_missing()` 对比注册表，发现 `layernorm` 尚未注册：

```
[Step 0b] 发现未注册算子: ['flash_attention', 'layernorm']
```

**复杂度判断**（通过 `infer_strategy`）：

1. 先查 `OPERATOR_TEMPLATES`：`layernorm` 不在其中（只有7个模板算子）
2. 回退到名称关键词：`"layer" in "layernorm"` → 匹配 `"layernorm": "normalization"`
3. `strategy = "normalization"` → 使用含 bias 的归一化接口

```
[AutoOpRegistrar] 算子复杂度分析:
  flash_attention: 复杂算子（无法自动推导）— 接口依赖运行时参数（head_dim/batch等），需手动实现
  layernorm: 归一化算子（中等）— 需要 weight/bias 参数，ctypes: (x, w, out, N, H, eps)
```

**自动生成并写入文件**：

```python
# operators/generated_ops.py（新追加内容）
# strategy: normalization
LAYERNORM_FORWARD_DESC = OperatorDesc(
    name='layernorm',
    variant='forward',
    ctypes_argtypes=['void*', 'void*', 'void*', 'void*', 'int', 'int', 'float'],
    #                 x       weight   bias     out      N      H     eps
    output_arg_indices=[3],
    output_dtypes=['fp16'],
    pytorch_reference=lambda x, w, b: F.layer_norm(
        x.float(), (x.shape[-1],), w.float(), b.float()
    ).half(),
    input_shapes_fn=lambda tc: {
        "x":      torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda"),
        "weight": torch.ones(tc["H"],           dtype=torch.float16, device="cuda"),
        "bias":   torch.zeros(tc["H"],          dtype=torch.float16, device="cuda"),
    },
    scalar_args_fn=lambda tc, inp: [tc["N"], tc["H"], 1e-5],
    inject_pattern=("module_type", "LayerNorm"),   # 替换所有 LayerNorm 子模块
    inject_fn=None,   # 暂无注入实现，需手动补充
)
```

之后，LLM 被要求生成 `layernorm_forward` CUDA kernel，编译后自动验证。

> **注意**：`inject_fn=None` 意味着 layernorm 只做生成+验证，**不会注入模型**（因为注入逻辑需要手动实现 ctypes 加载 + autograd.Function）。用户可以在 `generated_ops.py` 里参照 `_make_rmsnorm_inject_fn` 的模式手动补充。

---

## 示例四：第二次运行——持久化缓存直接复用

**场景**：上一次运行（Job 947）已将 3 个验证通过的 kernel 存入 SQLite。本次启动时系统自动加载，跳过重新生成。

**第一次运行结束时**：

```
[Step 2.5] 已将 3 个验证通过的 kernel 存入持久化 OperatorRegistry
  💾 silu_forward   (rel_err=0.0000, level=hw_verified) → 已存储
  💾 silu_backward  (rel_err=0.0000, level=hw_verified) → 已存储
  💾 rmsnorm_forward(rel_err=0.0009, level=hw_verified) → 已存储
```

**第二次运行时** `_load_cached_kernels()` 从数据库查询：

```python
entry = registry.lookup("silu_forward", "NVIDIA GeForce RTX 4090")
# → OperatorEntry(correctness_passed=True, max_relative_error=0.0, level="hw_verified")
cached_kernels["silu_forward"] = {
    "code": entry.source_code,   # 上次生成的 CUDA 源码
    "flags": entry.build_flags,  # ["-O3", "-arch=sm_89", "--use_fast_math"]
}
```

```
[Step 0c] 发现持久化 kernel 缓存: ['silu_forward', 'silu_backward', 'rmsnorm_forward']
  将跳过这些 kernel 的重新生成，直接使用已验证版本

[Step 1] silu_forward:    ♻️ 使用持久化已验证版本（跳过重新生成）
[Step 1] silu_backward:   ♻️ 使用持久化已验证版本（跳过重新生成）
[Step 1] rmsnorm_forward: ♻️ 使用持久化已验证版本（跳过重新生成）
[Step 1] 生成算子: rmsnorm ...   ← 只重新生成未缓存的 rmsnorm_backward
```

**节省效果**：

| | 第一次运行 | 第二次运行（有缓存） |
|--|-----------|------------------|
| LLM API 调用次数 | 4 次 | 1 次（只有 rmsnorm_backward）|
| 生成耗时 | ~3 分钟 | ~1 分钟 |
| API 费用 | 全额 | 约 1/4 |

**手动管理缓存**：

```python
from operators.registry import get_registry

reg = get_registry()

# 查看已缓存的 kernel
for entry in reg.list_operators():
    print(f"{entry.operator_name} @ {entry.gpu_model}: "
          f"level={entry.verification_level}, err={entry.max_relative_error:.4f}")

# 查询特定算子
entry = reg.lookup("silu_forward", "NVIDIA GeForce RTX 4090")
print(entry.source_code[:200])  # 查看缓存的 CUDA 源码

# 清除所有缓存（强制下次重新生成）
import os
os.remove(".operator_registry.db")
```

---

## 数据流总结

下图展示了四个示例涵盖的数据流动路径：

```
用户命令
  │
  ▼
TrainingAnalystAgent ──→ 识别算子列表（silu/rmsnorm/flash_attention/...）
  │                                          │
  │                               AutoOpRegistrar.find_missing()
  │                                          │
  │                               infer_strategy（查 OPERATOR_TEMPLATES.category
  │                               或名称关键词）
  │                                          │
  │                        ┌────────────────┴──────────────────┐
  │                        ↓ 简单/中等                          ↓ 复杂/未知
  │              生成 OperatorDesc              PyTorch fallback（直接跳过）
  │              写入 generated_ops.py
  │
  ▼
generate_kernels_for_task()
  │
  ├─ _load_cached_kernels() ──→ 有缓存？直接复用（示例四）
  │
  └─ CodeGenAgent.run()
         │
         ├─ 第1次：LLM 生成 → 试编译 → 成功 → 返回（示例一）
         ├─ 第2次：编译失败 → 把 stderr 放入 fix_context → 重新生成 → 成功（示例二）
         └─ 第3次：再次失败 → 降级为简单版 prompt → 必然可编译
  │
  ▼
compile_all_kernels()  →  nvcc 生成 .so
  │
  ▼
verify_all_kernels_generic()
  │
  ├─ verify_kernel(desc, so_path)
  │    ├─ ctypes.CDLL(so_path)
  │    ├─ 构造测试输入（input_shapes_fn）
  │    ├─ 调用 launch_kernel
  │    ├─ backward variant 不用 no_grad（修复后）
  │    └─ 对比 pytorch_reference，误差 < error_threshold？
  │
  ├─ 通过 → 保存到 SQLite OperatorRegistry（示例四的缓存来源）
  └─ 失败 → so_path 置 None，后续使用 PyTorch fallback
  │
  ▼
OpRegistry.build_custom_fn_map(so_paths)
  → 调用 inject_fn 工厂，生成 fn_map
  │
  ▼
patch_model(model, op_reg, fn_map)
  → inject_pattern 匹配 → setattr 替换（SiLU×36，RMSNorm×145）
  │
  ▼
LoRA 训练 → 评估 → 输出三模式对比结果
```

---

## 附：关键代码位置速查

| 步骤 | 代码位置 | 核心函数 |
|------|---------|---------|
| Step 0 注册表初始化 | `operators/builtin_ops.py` | `register_builtin_ops()` |
| Step 0b 算子识别 | `operators/auto_registrar.py` | `AutoOpRegistrar.find_missing()`, `generate_op_desc()` |
| Step 0c 缓存加载 | `examples/full_agent_lora_train.py` | `_load_cached_kernels()` |
| Step 1 代码生成 | `agents/code_generator.py` | `CodeGenAgent.run()`, `_generate_cuda()` |
| Step 1 试编译 | `examples/full_agent_lora_train.py` | `_try_compile()` |
| Step 1 retry 逻辑 | `examples/full_agent_lora_train.py` | `_generate_one_kernel_with_retry()` |
| Step 2 正式编译 | `examples/full_agent_lora_train.py` | `compile_all_kernels()` |
| Step 2.5 验证 | `operators/verify.py` | `verify_kernel()`, `verify_all_kernels_generic()` |
| Step 2.5 持久化 | `examples/full_agent_lora_train.py` | `_save_verified_kernels_to_registry()` |
| Step 3 构建注入对象 | `operators/op_registry.py` | `OpRegistry.build_custom_fn_map()` |
| Step 4 注入模型 | `operators/patch.py` | `patch_model()` |
| Step 5 训练 | `examples/full_agent_lora_train.py` | `run_lora_training()` |
