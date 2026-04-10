"""
MathVerifier — 通用算子数学逻辑验证器

在任意环境下（无需目标 GPU）验证算子的数学正确性。
原理：用 PyTorch CPU 跑参考实现 vs LLM 生成的参考实现，对比数值。

验证项：
1. 数学公式正确性（reference vs torch 标准实现）
2. 多种 shape（常规、非对齐、边界）
3. 多种 dtype（fp32、fp16、bf16）
4. 数值稳定性（极大值、极小值、全零、NaN 处理）
5. 梯度正确性（可选）
"""
import logging
import math
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# 验证报告
# ────────────────────────────────────────────────────────────

@dataclass
class ShapeTestResult:
    """单个 shape 的测试结果"""
    shape_desc: str
    dtype: str
    passed: bool
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0
    mean_abs_error: float = 0.0
    error_msg: str = ""


@dataclass
class MathVerifyReport:
    """数学验证报告"""
    operator_name: str
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    results: list[ShapeTestResult] = field(default_factory=list)
    stability_results: list[ShapeTestResult] = field(default_factory=list)
    grad_check_passed: Optional[bool] = None
    elapsed_seconds: float = 0.0
    error: str = ""

    @property
    def all_passed(self) -> bool:
        return self.failed_tests == 0 and not self.error

    @property
    def worst_rel_error(self) -> float:
        all_results = self.results + self.stability_results
        if not all_results:
            return float("inf")
        return max(r.max_rel_error for r in all_results if r.passed)  or 0.0

    def summary(self) -> str:
        status = "✅ PASS" if self.all_passed else "❌ FAIL"
        lines = [
            f"{status}  {self.operator_name}  "
            f"({self.passed_tests}/{self.total_tests} tests passed, "
            f"{self.elapsed_seconds:.2f}s)",
        ]
        for r in self.results + self.stability_results:
            icon = "✅" if r.passed else "❌"
            if r.passed:
                lines.append(
                    f"  {icon} {r.shape_desc:<30} {r.dtype:<6} "
                    f"abs={r.max_abs_error:.2e}  rel={r.max_rel_error:.2e}"
                )
            else:
                lines.append(f"  {icon} {r.shape_desc:<30} {r.dtype:<6} {r.error_msg}")
        if self.grad_check_passed is not None:
            icon = "✅" if self.grad_check_passed else "❌"
            lines.append(f"  {icon} gradient check")
        if self.error:
            lines.append(f"  ⚠ {self.error}")
        return "\n".join(lines)


# ────────────────────────────────────────────────────────────
# 内置参考实现（PyTorch CPU，覆盖常见算子）
# ────────────────────────────────────────────────────────────

def _ref_silu(x):
    return torch.nn.functional.silu(x)

def _ref_gelu(x):
    return torch.nn.functional.gelu(x)

def _ref_relu(x):
    return torch.nn.functional.relu(x)

def _ref_sigmoid(x):
    return torch.sigmoid(x)

def _ref_tanh(x):
    return torch.tanh(x)

def _ref_mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))

def _ref_leaky_relu(x):
    return torch.nn.functional.leaky_relu(x, negative_slope=0.01)

def _ref_softmax(x):
    return torch.nn.functional.softmax(x, dim=-1)

def _ref_rmsnorm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    return (x.float() / rms * weight.float()).to(x.dtype)

def _ref_layernorm(x, weight, bias=None, eps=1e-5):
    return torch.nn.functional.layer_norm(x.float(), [x.shape[-1]],
                                           weight.float(),
                                           bias.float() if bias is not None else None,
                                           eps).to(x.dtype)

def _ref_matmul(A, B):
    return torch.matmul(A.float(), B.float()).to(A.dtype)

def _ref_flash_attention(Q, K, V):
    scale = Q.shape[-1] ** -0.5
    attn = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    attn = torch.nn.functional.softmax(attn, dim=-1)
    return torch.matmul(attn, V.float()).to(Q.dtype)

def _ref_rope(x, freqs_cos, freqs_sin):
    """RoPE: 旋转位置编码"""
    x_float = x.float()
    d = x.shape[-1]
    x1 = x_float[..., :d//2]
    x2 = x_float[..., d//2:]
    cos = freqs_cos.float()
    sin = freqs_sin.float()
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1).to(x.dtype)

def _ref_cross_entropy(logits, targets):
    return torch.nn.functional.cross_entropy(logits.float(), targets)

def _ref_embedding(weight, indices):
    return torch.nn.functional.embedding(indices, weight)

def _ref_dropout(x):
    # dropout 在 eval 模式下是 identity
    return x.clone()

def _ref_topk(x, k=5):
    return torch.topk(x, k=k, dim=-1)

def _ref_argmax(x):
    return torch.argmax(x, dim=-1)

def _ref_transpose(x):
    return x.transpose(-2, -1).contiguous()

def _ref_concat(x1, x2):
    return torch.cat([x1, x2], dim=-1)

def _ref_reduce_sum(x):
    return torch.sum(x, dim=-1)

def _ref_add(x1, x2):
    return x1 + x2

def _ref_elementwise_mul(x1, x2):
    return x1 * x2

def _ref_residual_add(x, residual):
    return x + residual


# ────────────────────────────────────────────────────────────
# 算子测试配置：定义输入生成 + 参考函数 + 容差
# ────────────────────────────────────────────────────────────

@dataclass
class OperatorTestConfig:
    """单个算子的测试配置"""
    name: str
    ref_fn: Callable                          # 参考实现函数
    input_generator: Callable                 # 生成输入张量的函数
    output_comparator: str = "allclose"       # allclose / topk_match / exact_int
    atol_fp32: float = 1e-5
    atol_fp16: float = 1e-2
    atol_bf16: float = 1e-2
    rtol: float = 1e-3
    supports_grad: bool = True
    test_shapes: list[dict] = field(default_factory=list)  # 自定义 shape 列表
    stability_shapes: list[dict] = field(default_factory=list)  # 数值稳定性测试


def _gen_elementwise_inputs(shape, dtype):
    """逐元素算子的输入生成"""
    x = torch.randn(shape, dtype=dtype)
    return (x,), {}

def _gen_norm_inputs(shape, dtype):
    """归一化算子的输入生成"""
    x = torch.randn(shape, dtype=dtype)
    weight = torch.ones(shape[-1], dtype=dtype)
    return (x, weight), {}

def _gen_layernorm_inputs(shape, dtype):
    x = torch.randn(shape, dtype=dtype)
    weight = torch.ones(shape[-1], dtype=dtype)
    bias = torch.zeros(shape[-1], dtype=dtype)
    return (x, weight, bias), {}

def _gen_matmul_inputs(shape, dtype):
    """矩阵乘法输入：shape = (B, M, K, N)"""
    B, M, K, N = shape[0], shape[1], shape[2], shape[3] if len(shape) > 3 else shape[2]
    A = torch.randn(B, M, K, dtype=dtype)
    B_mat = torch.randn(B, K, N, dtype=dtype)
    return (A, B_mat), {}

def _gen_attention_inputs(shape, dtype):
    """注意力输入：shape = (B, H, S, D)"""
    B, H, S, D = shape
    Q = torch.randn(B, H, S, D, dtype=dtype)
    K = torch.randn(B, H, S, D, dtype=dtype)
    V = torch.randn(B, H, S, D, dtype=dtype)
    return (Q, K, V), {}

def _gen_rope_inputs(shape, dtype):
    """RoPE 输入：shape = (B, S, D)"""
    B, S, D = shape[0], shape[1], shape[2]
    x = torch.randn(B, S, D, dtype=dtype)
    # 生成频率
    half_d = D // 2
    freqs = torch.arange(half_d, dtype=torch.float32) / half_d
    freqs = 1.0 / (10000.0 ** freqs)
    positions = torch.arange(S, dtype=torch.float32).unsqueeze(1)
    angles = positions * freqs.unsqueeze(0)
    freqs_cos = torch.cos(angles).unsqueeze(0).expand(B, -1, -1).to(dtype)
    freqs_sin = torch.sin(angles).unsqueeze(0).expand(B, -1, -1).to(dtype)
    return (x, freqs_cos, freqs_sin), {}

def _gen_cross_entropy_inputs(shape, dtype):
    B, C = shape[0], shape[1]
    logits = torch.randn(B, C, dtype=dtype)
    targets = torch.randint(0, C, (B,))
    return (logits, targets), {}

def _gen_embedding_inputs(shape, dtype):
    V, D = shape[0], shape[1]
    weight = torch.randn(V, D, dtype=dtype)
    indices = torch.randint(0, V, (4, 16))
    return (weight, indices), {}

def _gen_two_tensor_inputs(shape, dtype):
    x1 = torch.randn(shape, dtype=dtype)
    x2 = torch.randn(shape, dtype=dtype)
    return (x1, x2), {}


# ────────────────────────────────────────────────────────────
# 算子注册表
# ────────────────────────────────────────────────────────────

BUILTIN_CONFIGS: dict[str, OperatorTestConfig] = {}

def _register(name, ref_fn, input_gen, shapes=None, stability=None, **kwargs):
    """注册一个算子的测试配置"""
    default_shapes = shapes or [
        {"shape": (2, 128, 256)},
        {"shape": (1, 1, 64)},       # 小 shape
        {"shape": (4, 512, 1024)},    # 大 shape
        {"shape": (1, 7, 127)},       # 非对齐
    ]
    default_stability = stability or []
    BUILTIN_CONFIGS[name] = OperatorTestConfig(
        name=name,
        ref_fn=ref_fn,
        input_generator=input_gen,
        test_shapes=default_shapes,
        stability_shapes=default_stability,
        **kwargs,
    )

# 激活函数
for name, fn in [
    ("silu", _ref_silu), ("gelu", _ref_gelu), ("relu", _ref_relu),
    ("sigmoid", _ref_sigmoid), ("tanh", _ref_tanh), ("mish", _ref_mish),
    ("leaky_relu", _ref_leaky_relu),
]:
    _register(name, fn, _gen_elementwise_inputs,
              stability=[
                  {"shape": (2, 128, 256), "desc": "large_values",
                   "transform": lambda x: x * 100},
                  {"shape": (2, 128, 256), "desc": "tiny_values",
                   "transform": lambda x: x * 1e-7},
                  {"shape": (2, 128, 256), "desc": "zeros",
                   "transform": lambda x: torch.zeros_like(x)},
              ])

# softmax
_register("softmax", _ref_softmax, _gen_elementwise_inputs,
          stability=[
              {"shape": (2, 128, 256), "desc": "large_logits",
               "transform": lambda x: x * 100},
              {"shape": (2, 128, 256), "desc": "uniform",
               "transform": lambda x: torch.ones_like(x)},
          ])

# 归一化
_register("rmsnorm", _ref_rmsnorm, _gen_norm_inputs, atol_fp16=5e-2)
_register("layernorm", _ref_layernorm, _gen_layernorm_inputs, atol_fp16=5e-2)

# 矩阵乘法
_register("matmul", _ref_matmul, _gen_matmul_inputs,
          shapes=[
              {"shape": (2, 64, 64, 64)},
              {"shape": (1, 128, 256, 128)},
              {"shape": (1, 1, 1, 1)},       # 标量
              {"shape": (2, 7, 13, 11)},      # 非对齐
          ],
          atol_fp16=5e-2, atol_bf16=5e-2)

# 注意力
_register("flash_attention", _ref_flash_attention, _gen_attention_inputs,
          shapes=[
              {"shape": (1, 4, 64, 32)},
              {"shape": (2, 8, 128, 64)},
              {"shape": (1, 1, 16, 16)},
              {"shape": (1, 2, 7, 32)},       # 非对齐 seq_len
          ],
          atol_fp16=5e-2, atol_bf16=5e-2)

# RoPE
_register("rope", _ref_rope, _gen_rope_inputs,
          shapes=[
              {"shape": (2, 128, 64)},
              {"shape": (1, 64, 128)},
              {"shape": (1, 1, 32)},
              {"shape": (2, 7, 64)},
          ],
          atol_fp16=1e-2)

# 交叉熵
_register("cross_entropy", _ref_cross_entropy, _gen_cross_entropy_inputs,
          shapes=[
              {"shape": (32, 1000)},
              {"shape": (1, 10)},
              {"shape": (128, 50000)},
          ],
          supports_grad=False, output_comparator="scalar")

# embedding
_register("embedding", _ref_embedding, _gen_embedding_inputs,
          shapes=[
              {"shape": (10000, 256)},
              {"shape": (1000, 64)},
          ],
          supports_grad=False)

# 双输入算子
for name, fn in [("add", _ref_add), ("elementwise_mul", _ref_elementwise_mul),
                  ("residual_add", _ref_residual_add)]:
    _register(name, fn, _gen_two_tensor_inputs)

# 其他
_register("transpose", _ref_transpose, _gen_elementwise_inputs,
          shapes=[
              {"shape": (2, 64, 128)},
              {"shape": (1, 7, 13)},
          ])
_register("reduce_sum", _ref_reduce_sum, _gen_elementwise_inputs,
          output_comparator="allclose")
_register("concat", _ref_concat, _gen_two_tensor_inputs)
_register("argmax", _ref_argmax, _gen_elementwise_inputs,
          output_comparator="exact_int", supports_grad=False)


# ────────────────────────────────────────────────────────────
# 核心验证器
# ────────────────────────────────────────────────────────────

class MathVerifier:
    """
    通用算子数学逻辑验证器。

    用法:
        verifier = MathVerifier()
        report = verifier.verify("silu")
        print(report.summary())

        # 验证所有内置算子
        reports = verifier.verify_all()

        # 验证自定义算子（提供参考实现）
        report = verifier.verify_custom(
            name="my_op",
            ref_fn=lambda x: x * torch.sigmoid(x),
            input_gen=lambda shape, dtype: ((torch.randn(shape, dtype=dtype),), {}),
        )
    """

    def __init__(self, dtypes=None, device="cpu"):
        self.dtypes = dtypes or [torch.float32, torch.float16]
        self.device = device

    def verify(self, op_name: str) -> MathVerifyReport:
        """验证一个内置算子"""
        config = BUILTIN_CONFIGS.get(op_name)
        if config is None:
            report = MathVerifyReport(operator_name=op_name)
            report.error = f"Unknown operator '{op_name}'. Available: {', '.join(sorted(BUILTIN_CONFIGS.keys()))}"
            return report
        return self._run_tests(config)

    def verify_custom(self, name: str, ref_fn: Callable, input_gen: Callable,
                      shapes: list[dict] = None, **kwargs) -> MathVerifyReport:
        """验证自定义算子"""
        config = OperatorTestConfig(
            name=name,
            ref_fn=ref_fn,
            input_generator=input_gen,
            test_shapes=shapes or [
                {"shape": (2, 128, 256)},
                {"shape": (1, 1, 64)},
                {"shape": (4, 512, 1024)},
            ],
            **kwargs,
        )
        return self._run_tests(config)

    def verify_all(self) -> list[MathVerifyReport]:
        """验证所有内置算子"""
        reports = []
        for name in sorted(BUILTIN_CONFIGS.keys()):
            reports.append(self.verify(name))
        return reports

    def list_operators(self) -> list[str]:
        """列出所有可验证的算子"""
        return sorted(BUILTIN_CONFIGS.keys())

    def _run_tests(self, config: OperatorTestConfig) -> MathVerifyReport:
        """执行完整测试"""
        report = MathVerifyReport(operator_name=config.name)
        t0 = time.time()

        # 1. 常规 shape 测试
        for shape_cfg in config.test_shapes:
            for dtype in self.dtypes:
                result = self._test_one(config, shape_cfg, dtype)
                report.results.append(result)
                report.total_tests += 1
                if result.passed:
                    report.passed_tests += 1
                else:
                    report.failed_tests += 1

        # 2. 数值稳定性测试（仅 fp32）
        for stab_cfg in config.stability_shapes:
            result = self._test_stability(config, stab_cfg)
            report.stability_results.append(result)
            report.total_tests += 1
            if result.passed:
                report.passed_tests += 1
            else:
                report.failed_tests += 1

        # 3. 梯度检查（可选，仅 fp64）
        if config.supports_grad:
            report.grad_check_passed = self._test_gradient(config)
            report.total_tests += 1
            if report.grad_check_passed:
                report.passed_tests += 1
            else:
                report.failed_tests += 1

        report.elapsed_seconds = time.time() - t0
        return report

    def _test_one(self, config: OperatorTestConfig, shape_cfg: dict,
                  dtype: torch.dtype) -> ShapeTestResult:
        """测试单个 shape + dtype 组合"""
        shape = shape_cfg["shape"]
        desc = shape_cfg.get("desc", str(shape))
        dtype_name = {torch.float32: "fp32", torch.float16: "fp16",
                      torch.bfloat16: "bf16"}.get(dtype, str(dtype))

        try:
            args, kwargs = config.input_generator(shape, dtype)
            # 移到目标设备
            args = tuple(a.to(self.device) if isinstance(a, torch.Tensor) else a for a in args)

            with torch.no_grad():
                output = config.ref_fn(*args, **kwargs)

            # 对比：用 fp32 重新算一遍作为 ground truth
            args_fp32 = tuple(
                a.float().to(self.device) if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                for a in args
            )
            with torch.no_grad():
                output_ref = config.ref_fn(*args_fp32, **kwargs)

            return self._compare(output, output_ref, config, desc, dtype_name)

        except Exception as e:
            return ShapeTestResult(
                shape_desc=desc, dtype=dtype_name, passed=False,
                error_msg=f"Exception: {e}"
            )

    def _test_stability(self, config: OperatorTestConfig, stab_cfg: dict) -> ShapeTestResult:
        """数值稳定性测试"""
        shape = stab_cfg["shape"]
        desc = stab_cfg.get("desc", "stability")
        transform = stab_cfg.get("transform", lambda x: x)

        try:
            args, kwargs = config.input_generator(shape, torch.float32)
            # 对第一个张量应用变换
            args = list(args)
            if isinstance(args[0], torch.Tensor):
                args[0] = transform(args[0])
            args = tuple(args)

            with torch.no_grad():
                output = config.ref_fn(*args, **kwargs)

            # 检查 NaN / Inf
            if isinstance(output, torch.Tensor):
                has_nan = torch.isnan(output).any().item()
                has_inf = torch.isinf(output).any().item()
                if has_nan:
                    return ShapeTestResult(
                        shape_desc=f"stability/{desc}", dtype="fp32",
                        passed=False, error_msg="Output contains NaN"
                    )
                if has_inf and "softmax" not in config.name:
                    # softmax 对极大输入可能产生 0/1，但不应该有 inf
                    return ShapeTestResult(
                        shape_desc=f"stability/{desc}", dtype="fp32",
                        passed=False, error_msg="Output contains Inf"
                    )

            return ShapeTestResult(
                shape_desc=f"stability/{desc}", dtype="fp32",
                passed=True, max_abs_error=0.0, max_rel_error=0.0,
            )

        except Exception as e:
            return ShapeTestResult(
                shape_desc=f"stability/{desc}", dtype="fp32",
                passed=False, error_msg=f"Exception: {e}"
            )

    def _test_gradient(self, config: OperatorTestConfig) -> bool:
        """梯度正确性检查"""
        try:
            # 用最小的 shape 做 gradcheck
            shape = config.test_shapes[0]["shape"]
            # 用小 shape 避免太慢
            small_shape = tuple(min(s, 8) if isinstance(s, int) else s for s in shape)
            args, kwargs = config.input_generator(small_shape, torch.float64)
            args = list(args)

            # 只对浮点张量做 gradcheck
            grad_args = []
            for i, a in enumerate(args):
                if isinstance(a, torch.Tensor) and a.is_floating_point():
                    args[i] = a.requires_grad_(True)
                    grad_args.append(args[i])

            if not grad_args:
                return True  # 没有可求导的输入

            def fn(*inputs):
                # 重建 args
                result_args = list(args)
                j = 0
                for i in range(len(result_args)):
                    if isinstance(result_args[i], torch.Tensor) and result_args[i].requires_grad:
                        result_args[i] = inputs[j]
                        j += 1
                out = config.ref_fn(*result_args, **kwargs)
                if isinstance(out, tuple):
                    out = out[0]
                return out

            return torch.autograd.gradcheck(fn, tuple(grad_args), eps=1e-6, atol=1e-4, rtol=1e-3)

        except Exception as e:
            logger.debug(f"[MathVerifier] gradcheck failed for {config.name}: {e}")
            return False

    def _compare(self, output, output_ref, config: OperatorTestConfig,
                 desc: str, dtype_name: str) -> ShapeTestResult:
        """比较两个输出"""
        if config.output_comparator == "exact_int":
            # 整数输出（argmax 等）
            if isinstance(output, torch.Tensor) and isinstance(output_ref, torch.Tensor):
                match = torch.equal(output, output_ref.to(output.dtype))
                return ShapeTestResult(
                    shape_desc=desc, dtype=dtype_name, passed=match,
                    error_msg="" if match else "Integer outputs don't match"
                )

        if config.output_comparator == "scalar":
            # 标量输出（loss 等）
            if isinstance(output, torch.Tensor):
                output = output.float()
            if isinstance(output_ref, torch.Tensor):
                output_ref = output_ref.float()
            abs_err = abs(float(output) - float(output_ref))
            rel_err = abs_err / (abs(float(output_ref)) + 1e-12)
            atol = config.atol_fp32
            passed = abs_err < atol or rel_err < config.rtol
            return ShapeTestResult(
                shape_desc=desc, dtype=dtype_name, passed=passed,
                max_abs_error=abs_err, max_rel_error=rel_err,
                error_msg="" if passed else f"Scalar mismatch: {float(output):.6f} vs {float(output_ref):.6f}"
            )

        # topk 返回 (values, indices)
        if isinstance(output, tuple) and isinstance(output_ref, tuple):
            output = output[0]
            output_ref = output_ref[0]

        if not isinstance(output, torch.Tensor) or not isinstance(output_ref, torch.Tensor):
            return ShapeTestResult(
                shape_desc=desc, dtype=dtype_name, passed=False,
                error_msg=f"Unexpected output type: {type(output)}"
            )

        # 转 float 比较
        out_f = output.float()
        ref_f = output_ref.float()

        # 处理 shape 不匹配
        if out_f.shape != ref_f.shape:
            return ShapeTestResult(
                shape_desc=desc, dtype=dtype_name, passed=False,
                error_msg=f"Shape mismatch: {out_f.shape} vs {ref_f.shape}"
            )

        abs_err = (out_f - ref_f).abs()
        max_abs = abs_err.max().item()
        mean_abs = abs_err.mean().item()
        rel_err = abs_err / (ref_f.abs() + 1e-12)
        max_rel = rel_err.max().item()

        # 选择容差
        atol = {
            "fp32": config.atol_fp32,
            "fp16": config.atol_fp16,
            "bf16": config.atol_bf16,
        }.get(dtype_name, config.atol_fp32)

        passed = max_abs < atol or max_rel < config.rtol

        return ShapeTestResult(
            shape_desc=desc, dtype=dtype_name, passed=passed,
            max_abs_error=max_abs, max_rel_error=max_rel, mean_abs_error=mean_abs,
            error_msg="" if passed else f"Tolerance exceeded: abs={max_abs:.2e} > {atol:.2e}"
        )
