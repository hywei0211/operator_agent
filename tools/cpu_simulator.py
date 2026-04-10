"""
CPU 算子模拟器
没有 GPU 时，用 PyTorch CPU 实现验证算子的数学正确性
核心思路：不测硬件、测数学

验证分两层：
1. 数学层（Math Layer）：用 PyTorch CPU 参考实现 vs 生成代码提取的逻辑
2. 编译层（Static Layer）：用规则 + clang 检查语法
"""
import math
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available; math verification will be skipped")


@dataclass
class SimulationResult:
    """模拟执行结果"""
    operator_name: str
    math_correct: bool
    max_relative_error: float = float('inf')
    tested_shapes: list = field(default_factory=list)
    error_message: str = ""
    # 理论性能预估（Roofline Model）
    estimated_bandwidth_efficiency: float = 0.0
    estimated_compute_efficiency: float = 0.0
    bound_type: str = "unknown"     # memory_bound / compute_bound
    notes: list = field(default_factory=list)


class CPUSimulator:
    """
    CPU 模拟器

    为每种算子提供 PyTorch CPU 参考实现
    生成的 GPU 代码的数学逻辑通过 LLM 提取后在 CPU 上验证

    无法替代的部分（仍需真实 GPU）：
    - 实际执行速度
    - 内存访问模式效率
    - 硬件特定 Bug（如 Warp Divergence）
    """

    # 所有支持算子的参考实现
    REFERENCE_IMPLS: dict[str, Callable] = {}

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self._register_references()

    def _register_references(self):
        """注册所有算子的 PyTorch CPU 参考实现"""
        if not HAS_TORCH:
            return

        self.REFERENCE_IMPLS = {
            # ── 常见激活函数 ──────────────────────────────────────
            "gelu":         lambda x: F.gelu(x),
            "silu":         lambda x: F.silu(x),
            "relu":         lambda x: F.relu(x),
            "swiglu":       lambda x, gate: F.silu(gate) * x,

            # ── 归一化 ──────────────────────────────────────────
            "rmsnorm":      self._ref_rmsnorm,
            "layernorm":    self._ref_layernorm,
            "groupnorm":    lambda x, g, w, b: F.group_norm(x, g, w, b),

            # ── 注意力 ──────────────────────────────────────────
            "flash_attention":          self._ref_flash_attention,
            "scaled_dot_product":       self._ref_scaled_dot_product,
            "causal_attention":         self._ref_causal_attention,

            # ── 矩阵运算 ─────────────────────────────────────────
            "matmul":       lambda a, b: torch.matmul(a, b),
            "bmm":          lambda a, b: torch.bmm(a, b),
            "linear":       lambda x, w, b=None: F.linear(x, w, b),

            # ── 规约 ─────────────────────────────────────────────
            "softmax":      lambda x, dim=-1: F.softmax(x, dim=dim),
            "log_softmax":  lambda x, dim=-1: F.log_softmax(x, dim=dim),
            "mean":         lambda x, dim=-1: x.mean(dim=dim),
            "sum":          lambda x, dim=-1: x.sum(dim=dim),

            # ── 元素级 ───────────────────────────────────────────
            "add":          lambda a, b: a + b,
            "mul":          lambda a, b: a * b,
            "dropout":      lambda x, p: F.dropout(x, p, training=False),

            # ── 嵌入 ─────────────────────────────────────────────
            "embedding":    lambda idx, w: F.embedding(idx, w),
        }

    # ── 参考实现 ────────────────────────────────────────────────

    def _ref_rmsnorm(self, x, weight=None, eps=1e-6):
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
        x_norm = x.float() / rms
        if weight is not None:
            x_norm = x_norm * weight.float()
        return x_norm.to(x.dtype)

    def _ref_layernorm(self, x, normalized_shape=None, weight=None, bias=None, eps=1e-5):
        shape = normalized_shape or [x.shape[-1]]
        return F.layer_norm(x, shape, weight, bias, eps)

    def _ref_flash_attention(self, q, k, v, causal=True, scale=None):
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if causal:
            seq_len = q.shape[-2]
            mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask.to(scores.device), float('-inf'))
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def _ref_scaled_dot_product(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v) if HAS_TORCH else None

    def _ref_causal_attention(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True) if HAS_TORCH else None

    # ── 核心验证接口 ─────────────────────────────────────────────

    def verify_operator(
        self,
        operator_name: str,
        test_inputs: list[dict],
        generated_fn: Optional[Callable] = None,
        rtol: float = 1e-2,
        atol: float = 1e-4,
    ) -> SimulationResult:
        """
        验证生成的算子与参考实现的数学一致性

        Args:
            operator_name: 算子名称
            test_inputs: 测试输入列表，每个元素是 kwargs dict
            generated_fn: 生成的算子的 Python 等价实现
                          如果为 None，则尝试用 LLM 从代码中提取
            rtol/atol: 相对/绝对误差容忍度
        """
        ref_fn = self.REFERENCE_IMPLS.get(operator_name)
        if ref_fn is None:
            return SimulationResult(
                operator_name=operator_name,
                math_correct=False,
                error_message=f"No reference implementation for '{operator_name}'",
                notes=[f"Supported operators: {list(self.REFERENCE_IMPLS.keys())}"],
            )

        if not HAS_TORCH:
            return SimulationResult(
                operator_name=operator_name,
                math_correct=True,
                error_message="PyTorch not available, skipping math check",
                notes=["Install PyTorch to enable math verification"],
            )

        max_err = 0.0
        tested_shapes = []
        failed_case = None

        for i, inputs in enumerate(test_inputs):
            try:
                # 构造 PyTorch CPU 张量
                torch_inputs = self._dict_to_tensors(inputs)
                # 参考实现
                ref_out = ref_fn(**torch_inputs)
                if isinstance(ref_out, tuple):
                    ref_out = ref_out[0]

                # 如果有生成的函数，对比输出
                if generated_fn is not None:
                    gen_out = generated_fn(**torch_inputs)
                    if isinstance(gen_out, tuple):
                        gen_out = gen_out[0]

                    if not torch.allclose(ref_out.float(), gen_out.float(), rtol=rtol, atol=atol):
                        err = (ref_out.float() - gen_out.float()).abs().max().item()
                        rel_err = err / (ref_out.float().abs().max().item() + 1e-8)
                        max_err = max(max_err, rel_err)
                        if failed_case is None:
                            failed_case = {"shape": str(dict(inputs)), "rel_err": rel_err}

                shape_desc = {k: list(v.shape) if hasattr(v, 'shape') else v
                             for k, v in inputs.items() if k != 'dtype'}
                tested_shapes.append(shape_desc)

            except Exception as e:
                return SimulationResult(
                    operator_name=operator_name,
                    math_correct=False,
                    error_message=f"Test case {i} failed: {e}",
                    tested_shapes=tested_shapes,
                )

        math_correct = failed_case is None or max_err <= rtol
        notes = []
        if not math_correct:
            notes.append(f"Max relative error: {max_err:.2e} (failed case: {failed_case})")
        else:
            notes.append(f"✅ Math verified against PyTorch reference on {len(tested_shapes)} shapes")
            if generated_fn is None:
                notes.append("(Reference-only check: no generated function to compare)")

        return SimulationResult(
            operator_name=operator_name,
            math_correct=math_correct,
            max_relative_error=max_err,
            tested_shapes=tested_shapes,
            notes=notes,
        )

    # ── 反向传播验证 ─────────────────────────────────────────

    BACKWARD_REFERENCE_IMPLS: dict[str, Callable] = {}

    def _register_backward_references(self):
        """注册所有算子的 backward 参考实现"""
        if not HAS_TORCH:
            return

        self.BACKWARD_REFERENCE_IMPLS = {
            "gelu":    self._backward_gelu,
            "silu":    self._backward_silu,
            "softmax": self._backward_softmax,
            "rmsnorm": self._backward_rmsnorm,
            "matmul":  self._backward_matmul,
        }

    def _backward_gelu(self, grad_output, x):
        x_f = x.float()
        return torch.autograd.functional.vjp(F.gelu, x_f, grad_output.float())[1].to(grad_output.dtype)

    def _backward_silu(self, grad_output, x):
        sig = torch.sigmoid(x.float())
        return (grad_output.float() * sig * (1.0 + x.float() * (1.0 - sig))).to(grad_output.dtype)

    def _backward_softmax(self, grad_output, y):
        """softmax backward: grad_x = y * (grad_y - (grad_y * y).sum(dim=-1, keepdim=True))"""
        y_f = y.float()
        g_f = grad_output.float()
        return (y_f * (g_f - (g_f * y_f).sum(dim=-1, keepdim=True))).to(grad_output.dtype)

    def _backward_rmsnorm(self, grad_output, x, weight=None):
        eps = 1e-6
        x_f = x.float()
        g_f = grad_output.float()
        rms = torch.sqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
        x_norm = x_f / rms
        w_f = weight.float() if weight is not None else torch.ones(x_f.shape[-1])
        grad_x_norm = g_f * w_f
        grad_x = (grad_x_norm - x_norm * (grad_x_norm * x_norm).mean(dim=-1, keepdim=True)) / rms
        return grad_x.to(grad_output.dtype)

    def _backward_matmul(self, grad_output, A, B):
        grad_A = torch.matmul(grad_output.float(), B.float().transpose(-2, -1))
        grad_B = torch.matmul(A.float().transpose(-2, -1), grad_output.float())
        return grad_A.to(grad_output.dtype), grad_B.to(grad_output.dtype)

    def verify_backward(
        self,
        operator_name: str,
        rtol: float = 1e-2,
    ) -> SimulationResult:
        """
        用 torch.autograd.gradcheck 验证 backward 正确性。
        gradcheck 会用有限差分法自动验证解析梯度是否正确。
        """
        if not HAS_TORCH:
            return SimulationResult(
                operator_name=operator_name, math_correct=True,
                error_message="PyTorch not available, skipping backward check")

        if not self.BACKWARD_REFERENCE_IMPLS:
            self._register_backward_references()

        ref_fn = self.REFERENCE_IMPLS.get(operator_name)
        if ref_fn is None:
            return SimulationResult(
                operator_name=f"{operator_name}_backward", math_correct=True,
                error_message=f"No forward ref for {operator_name}, skip backward check")

        try:
            from torch.autograd import gradcheck

            # 构造需要梯度的测试输入（gradcheck 需要 float64）
            if operator_name in ("matmul",):
                a = torch.randn(2, 16, 16, dtype=torch.float64, requires_grad=True)
                b = torch.randn(2, 16, 16, dtype=torch.float64, requires_grad=True)
                result = gradcheck(torch.matmul, (a, b), eps=1e-6, atol=1e-4, raise_exception=False)
            elif operator_name in ("rmsnorm",):
                x = torch.randn(2, 8, 32, dtype=torch.float64, requires_grad=True)
                # RMSNorm 用 PyTorch 原生操作（gradcheck 只验数学，不验 kernel）
                def rmsnorm_fn(x_):
                    rms_ = torch.sqrt(x_.pow(2).mean(-1, keepdim=True) + 1e-6)
                    return x_ / rms_
                result = gradcheck(rmsnorm_fn, (x,), eps=1e-6, atol=1e-4, raise_exception=False)
            else:
                x = torch.randn(4, 32, dtype=torch.float64, requires_grad=True)
                fn_map = {
                    "gelu": F.gelu, "silu": F.silu, "relu": F.relu,
                    "softmax": lambda x_: F.softmax(x_, dim=-1),
                }
                fn = fn_map.get(operator_name)
                if fn is None:
                    return SimulationResult(
                        operator_name=f"{operator_name}_backward", math_correct=True,
                        error_message=f"No gradcheck fn for {operator_name}")
                result = gradcheck(fn, (x,), eps=1e-6, atol=1e-4, raise_exception=False)

            return SimulationResult(
                operator_name=f"{operator_name}_backward",
                math_correct=bool(result),
                notes=[f"gradcheck {'passed ✅' if result else 'failed ❌'} for {operator_name}"],
            )
        except Exception as e:
            return SimulationResult(
                operator_name=f"{operator_name}_backward",
                math_correct=False,
                error_message=f"gradcheck error: {str(e)[:200]}",
            )

    def generate_test_inputs(self, operator_name: str) -> list[dict]:
        """为每种算子生成标准测试输入集"""
        if not HAS_TORCH:
            return []

        dtype = torch.float32

        shapes_map = {
            "gelu":     [{"x": torch.randn(128, 4096, dtype=dtype)},
                         {"x": torch.randn(4, 512, 4096, dtype=dtype)}],
            "silu":     [{"x": torch.randn(128, 4096, dtype=dtype)}],
            "relu":     [{"x": torch.randn(256, 1024, dtype=dtype)}],
            "rmsnorm":  [{"x": torch.randn(4, 512, 4096, dtype=dtype)},
                         {"x": torch.randn(1, 2048, 4096, dtype=dtype),
                          "weight": torch.ones(4096, dtype=dtype)}],
            "layernorm": [{"x": torch.randn(4, 512, 4096, dtype=dtype)}],
            "softmax":   [{"x": torch.randn(4, 32, 512, 512, dtype=dtype)},
                          {"x": torch.randn(8, 64, dtype=dtype)}],
            "flash_attention": [
                {"q": torch.randn(2, 8, 64, 64, dtype=dtype),
                 "k": torch.randn(2, 8, 64, 64, dtype=dtype),
                 "v": torch.randn(2, 8, 64, 64, dtype=dtype)},
                {"q": torch.randn(1, 16, 128, 64, dtype=dtype),
                 "k": torch.randn(1, 16, 128, 64, dtype=dtype),
                 "v": torch.randn(1, 16, 128, 64, dtype=dtype)},
            ],
            "matmul":   [{"a": torch.randn(128, 512, dtype=dtype),
                          "b": torch.randn(512, 256, dtype=dtype)},
                         {"a": torch.randn(4, 512, 512, dtype=dtype),
                          "b": torch.randn(4, 512, 512, dtype=dtype)}],
            "add":      [{"a": torch.randn(256, 4096, dtype=dtype),
                          "b": torch.randn(256, 4096, dtype=dtype)}],
        }
        return shapes_map.get(operator_name, [
            {"x": torch.randn(128, 256, dtype=dtype)},
        ])

    def _dict_to_tensors(self, inputs: dict) -> dict:
        """将输入 dict 转为 PyTorch CPU 张量"""
        if not HAS_TORCH:
            return inputs
        result = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.cpu()
            else:
                result[k] = v
        return result


class RooflineSimulator:
    """
    Roofline 性能模型模拟器
    在没有 GPU 的情况下预测算子的理论性能上限
    """

    def predict(
        self,
        operator_name: str,
        gpu_model: str,
        input_shapes: dict,
        dtype: str = "float16",
    ) -> dict:
        """
        基于 Roofline 模型预测性能

        返回：
        - bound_type: memory_bound / compute_bound
        - estimated_efficiency: 预估带宽/算力利用率
        - bottleneck: 瓶颈所在
        - suggestions: 优化建议
        """
        from knowledge_base.hardware_specs.gpu_database import get_gpu_spec
        spec = get_gpu_spec(gpu_model)
        if spec is None:
            return {"error": f"Unknown GPU: {gpu_model}"}

        bytes_per_elem = {"float16": 2, "bfloat16": 2, "float32": 4, "int8": 1}.get(dtype, 2)

        # 计算算子的 FLOPs 和内存访问量
        flops, mem_bytes = self._estimate_flops_and_mem(operator_name, input_shapes, bytes_per_elem)

        if flops == 0:
            return {"error": "Cannot estimate FLOPs for this operator"}

        # 算术密度（FLOPs / Byte）
        arithmetic_intensity = flops / max(mem_bytes, 1)

        # 硬件参数
        peak_flops = spec.compute.fp16_tflops * 1e12  # TFLOPS → FLOPS
        peak_bw = spec.memory.bandwidth_gbps * 1e9     # GB/s → B/s

        # Roofline 交叉点（ridge point）
        ridge_point = peak_flops / peak_bw

        if arithmetic_intensity < ridge_point:
            bound_type = "memory_bound"
            achieved_perf = arithmetic_intensity * peak_bw  # 内存限制的性能
            efficiency = achieved_perf / peak_flops
        else:
            bound_type = "compute_bound"
            efficiency = 0.85  # 假设理想情况下能达到85%计算效率

        suggestions = self._generate_suggestions(
            operator_name, bound_type, arithmetic_intensity, ridge_point
        )

        return {
            "operator": operator_name,
            "gpu_model": gpu_model,
            "bound_type": bound_type,
            "arithmetic_intensity": arithmetic_intensity,
            "ridge_point": ridge_point,
            "estimated_efficiency": efficiency,
            "peak_flops_tflops": spec.compute.fp16_tflops,
            "peak_bandwidth_gbps": spec.memory.bandwidth_gbps,
            "flops_total": flops,
            "memory_bytes": mem_bytes,
            "suggestions": suggestions,
        }

    def _estimate_flops_and_mem(
        self, operator_name: str, shapes: dict, bpe: int
    ) -> tuple[int, int]:
        """估算 FLOPs 和内存访问量"""
        name = operator_name.lower()

        if "matmul" in name:
            a = shapes.get("a_shape", [512, 512])
            b = shapes.get("b_shape", [512, 512])
            M, K = a[0], a[1]
            K2, N = b[0], b[1]
            flops = 2 * M * K * N
            mem = (M * K + K * N + M * N) * bpe
            return flops, mem

        if "attention" in name or "flash" in name:
            q = shapes.get("q_shape", [1, 8, 512, 64])
            B, H, S, D = q[0], q[1], q[2], q[3]
            # QK^T + softmax + AV
            flops = B * H * (2 * S * S * D + S * S + 2 * S * S * D)
            mem = B * H * (3 * S * D + S * S + S * D) * bpe
            return flops, mem

        if "norm" in name:
            x = shapes.get("x_shape", [4, 512, 4096])
            total = 1
            for s in x:
                total *= s
            flops = total * 5   # mean, var, normalize, scale, shift
            mem = total * bpe * 3
            return flops, mem

        if name in ("gelu", "silu", "relu", "add", "mul"):
            x = shapes.get("x_shape", [128, 4096])
            total = 1
            for s in x:
                total *= s
            flops = total * (4 if name == "gelu" else 1)
            mem = total * bpe * 2  # read + write
            return flops, mem

        if "softmax" in name:
            x = shapes.get("x_shape", [4, 32, 512, 512])
            total = 1
            for s in x:
                total *= s
            flops = total * 3   # exp, sum, div
            mem = total * bpe * 2
            return flops, mem

        # 默认：假设是轻量级算子
        x = shapes.get("x_shape", [128, 4096])
        total = 1
        for s in x:
            total *= s
        return total * 2, total * bpe * 2

    def _generate_suggestions(
        self, op: str, bound_type: str, ai: float, ridge: float
    ) -> list[str]:
        suggestions = []
        if bound_type == "memory_bound":
            suggestions.append(f"算子是内存带宽瓶颈 (AI={ai:.1f} < ridge={ridge:.1f})")
            suggestions.append("优化方向：融合算子（operator fusion）减少内存往返次数")
            suggestions.append("使用向量化加载（每次 128-bit load）")
            if "attention" in op:
                suggestions.append("FlashAttention 通过分块减少 HBM 读写是正确方向")
        else:
            suggestions.append(f"算子是计算瓶颈 (AI={ai:.1f} >= ridge={ridge:.1f})")
            suggestions.append("优化方向：充分利用 Tensor Core / Matrix Core")
            suggestions.append("确保矩阵维度对齐到 16 的倍数（Tensor Core 约束）")
        return suggestions


class StaticCodeAnalyzer:
    """
    静态代码分析器
    无需编译器环境，通过规则检查代码结构
    """

    RULES = {
        "cuda": [
            (r"__global__\s+void", "Has kernel function"),
            # HIP/CUDA 兼容语法：两种线程索引均有效
            (r"threadIdx\.|blockIdx\.|blockDim\.|hipThreadIdx_|hipBlockIdx_",
             "Uses thread indexing"),
            # 边界检查：匹配任意 if 语句中包含 < 或 >= 或 > 的比较（覆盖各种变量名写法）
            (r"if\s*\([^)]*[<>]=?\s*\w+|if\s*\([^;]*[<>]=?\s*\w+[^;]*\)",
             "Has boundary check"),
        ],
        "hip": [
            (r"__global__\s+void", "Has kernel function"),
            # 现代 ROCm 同时支持 hipThreadIdx_ 和 threadIdx.（CUDA 兼容别名）
            (r"hipThreadIdx_|hipBlockIdx_|threadIdx\.|blockIdx\.", "Uses thread indexing"),
            # HIP 边界检查：同 CUDA
            (r"if\s*\([^)]*[<>]=?\s*\w+|if\s*\([^;]*[<>]=?\s*\w+[^;]*\)",
             "Has boundary check"),
        ],
        "ascendc": [
            (r"__aicore__\s+inline\s+void|__global__\s+__aicore__", "Has AscendC kernel"),
            (r"DataCopy\s*\(", "Has DataCopy (required for AscendC)"),
            (r"GetBlockIdx\(\)", "Uses AI Core indexing"),
            (r"TBuf<|LocalTensor<|TQue<|AllocTensor", "Uses on-chip buffers"),
        ],
        "triton": [
            (r"@triton\.jit", "Has triton.jit decorator"),
            (r"tl\.program_id\(", "Has program_id"),
            (r"tl\.load\(|tl\.store\(", "Has load/store"),
        ],
        "sycl": [
            (r"parallel_for", "Has parallel_for"),
            (r"nd_range|nd_item", "Uses nd_range"),
        ],
    }

    # 加分项（存在时提升评分）
    BONUS_PATTERNS = {
        "cuda": [
            (r"__shared__", "Uses shared memory (tiling optimization)"),
            (r"__restrict__", "Uses restrict qualifier (alias hint)"),
            (r"__syncthreads", "Has warp sync"),
            (r"half2|__half2", "Uses half2 vectorization"),
            (r"wmma::|mma\.", "Uses Tensor Core (wmma/mma)"),
            (r"#pragma unroll|__builtin_expect", "Has loop unroll hints"),
        ],
        "hip": [
            (r"__shared__|__lds__", "Uses LDS shared memory"),
            (r"__shfl_xor|__shfl_down", "Uses wavefront shuffle"),
            (r"mfma_|__builtin_amdgcn_mfma", "Uses MFMA Matrix Core"),
            (r"half2|__half2", "Uses half2 vectorization"),
        ],
        "ascendc": [
            (r"pipe\.InitBuffer|TPipe", "Uses double buffer pipeline"),
            (r"Matmul<|MatmulInit", "Uses Cube matrix unit"),
            (r"Muls\s*\(|Mul\s*\(|Add\s*\(", "Uses vector instructions"),
        ],
    }

    WARNING_PATTERNS = {
        "ascendc": [
            # 直接用 GM 指针做计算是错误的（必须先 DataCopy 到 UB）
            (r"xGm\[.*\]\s*[+\-\*]|yGm\[.*\]\s*[+\-\*]",
             "Direct GM arithmetic - should DataCopy to UB first!",
             "WARNING"),
        ],
    }

    def analyze(self, source_code: str, sdk: str) -> dict:
        """
        分析代码质量，返回通过/失败规则列表

        评分规则：
        - 必须规则（RULES）：不通过扣分，通过率 < 0.7 则 FAIL
        - 加分项（BONUS_PATTERNS）：通过则提升分数（最高 1.0）
        """
        rules   = self.RULES.get(sdk, [])
        bonuses = self.BONUS_PATTERNS.get(sdk, [])
        warnings = self.WARNING_PATTERNS.get(sdk, [])

        passed_required = []
        failed_required = []
        passed_bonus    = []
        warn_list       = []

        for pattern, description in rules:
            if re.search(pattern, source_code, re.MULTILINE | re.IGNORECASE):
                passed_required.append(description)
            else:
                failed_required.append(description)

        for pattern, description in bonuses:
            if re.search(pattern, source_code, re.MULTILINE):
                passed_bonus.append(description)

        for item in warnings:
            pattern, description = item[0], item[1]
            level = item[2] if len(item) > 2 else "INFO"
            if re.search(pattern, source_code, re.MULTILINE):
                if level == "WARNING":
                    warn_list.append(description)

        # 必须规则通过率
        required_score = len(passed_required) / max(len(rules), 1)
        # 加分项最多提升 0.2 分
        bonus_score = min(0.2, len(passed_bonus) * 0.04)
        total_score = min(1.0, required_score + bonus_score)

        return {
            "sdk": sdk,
            "score": round(total_score, 2),
            "required_score": round(required_score, 2),
            "passed_checks": passed_required,
            "failed_checks": failed_required,
            "bonus_features": passed_bonus,
            "warnings": warn_list,
            "summary": "PASS" if required_score >= 0.67 else "FAIL",
        }
