"""
内置算子描述注册
================
定义并注册以下算子的 OperatorDesc：
  - silu (forward + backward)
  - rmsnorm (forward + backward)
  - gelu (forward + backward)
  - matmul/linear (forward, 可注入 nn.Linear)
  - softmax (forward, 仅生成+验证，暂不注入)
  - cross_entropy (forward, 仅生成+验证，暂不注入)
  - embedding (forward, 仅生成+验证，暂不注入)

用法：
    from operators.builtin_ops import register_builtin_ops
    from operators.op_registry import get_op_registry
    reg = get_op_registry()
    register_builtin_ops(reg)
"""
from __future__ import annotations

import ctypes
import logging
import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.op_desc import OperatorDesc

if TYPE_CHECKING:
    from operators.op_registry import OpRegistry

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# inject_fn 工厂（复用 full_agent_lora_train.py 中的 ctypes 加载逻辑）
# ═══════════════════════════════════════════════════════════════════

def _make_silu_inject_fn(desc: OperatorDesc, so_paths: dict) -> dict:
    """
    SiLU inject_fn 工厂：
      给定 so_paths，加载 silu_forward.so 和 silu_backward.so，
      创建 SiLUCustomFunction + wrapper，返回 {"silu_fn": <callable>}。
    """
    forward_so = so_paths.get("silu_forward")
    backward_so = so_paths.get("silu_backward")

    forward_fn = None
    if forward_so and os.path.exists(forward_so):
        try:
            lib = ctypes.CDLL(forward_so)
            fn = lib.launch_kernel
            fn.restype = None
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            forward_fn = fn
            logger.info("[builtin_ops] ✅ SiLU forward kernel loaded")
        except (OSError, AttributeError) as e:
            logger.warning(f"[builtin_ops] SiLU forward load failed: {e}")

    backward_fn = None
    if backward_so and os.path.exists(backward_so):
        try:
            lib_bwd = ctypes.CDLL(backward_so)
            fn_bwd = lib_bwd.launch_kernel
            fn_bwd.restype = None
            fn_bwd.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_void_p, ctypes.c_int]
            backward_fn = fn_bwd
            logger.info("[builtin_ops] ✅ SiLU backward kernel loaded (float32 grad)")
        except (OSError, AttributeError) as e:
            logger.warning(f"[builtin_ops] SiLU backward load failed: {e}")

    class SiLUCustomFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            x_c = x.contiguous()
            if forward_fn is not None:
                out = torch.empty_like(x_c)
                forward_fn(x_c.data_ptr(), out.data_ptr(), x_c.numel())
                torch.cuda.synchronize()
            else:
                out = F.silu(x)
            ctx.save_for_backward(x)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            if backward_fn is not None:
                x_c = x.contiguous()
                g_c = grad_output.contiguous()
                grad_in_fp32 = torch.empty(
                    x_c.numel(), dtype=torch.float32, device=x_c.device
                )
                backward_fn(g_c.data_ptr(), x_c.data_ptr(),
                            grad_in_fp32.data_ptr(), x_c.numel())
                torch.cuda.synchronize()
                return grad_in_fp32.reshape(x_c.shape).to(x.dtype)
            else:
                sig = torch.sigmoid(x.float())
                return (grad_output.float() * sig * (1.0 + x.float() * (1.0 - sig))).to(x.dtype)

    def silu_custom(x):
        return SiLUCustomFunction.apply(x)

    return {"silu_fn": silu_custom}


def _make_rmsnorm_inject_fn(desc: OperatorDesc, so_paths: dict) -> dict:
    """
    RMSNorm inject_fn 工厂：
      创建 RMSNormCustomModule 类，返回 {"RMSNormModule": <class>}。
    """
    forward_so = so_paths.get("rmsnorm_forward")
    backward_so = so_paths.get("rmsnorm_backward")

    forward_fn = None
    if forward_so and os.path.exists(forward_so):
        try:
            lib = ctypes.CDLL(forward_so)
            fn = lib.launch_kernel
            fn.restype = None
            fn.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_float,
            ]
            forward_fn = fn
            logger.info("[builtin_ops] ✅ RMSNorm forward kernel loaded")
        except (OSError, AttributeError) as e:
            logger.warning(f"[builtin_ops] RMSNorm forward load failed: {e}")

    backward_fn = None
    if backward_so and os.path.exists(backward_so):
        try:
            lib_bwd = ctypes.CDLL(backward_so)
            fn_bwd = lib_bwd.launch_kernel
            fn_bwd.restype = None
            fn_bwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_float,
            ]
            backward_fn = fn_bwd
            logger.info("[builtin_ops] ✅ RMSNorm backward kernel loaded (float32 grad)")
        except (OSError, AttributeError) as e:
            logger.warning(f"[builtin_ops] RMSNorm backward load failed: {e}")

    class RMSNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, eps):
            x_c = x.contiguous()
            w_c = weight.contiguous()
            N = x_c.shape[0] if x_c.ndim > 1 else 1
            H = x_c.shape[-1]
            if forward_fn is not None:
                out = torch.empty_like(x_c)
                forward_fn(x_c.data_ptr(), w_c.data_ptr(), out.data_ptr(),
                           N, H, float(eps))
                torch.cuda.synchronize()
            else:
                x_fp = x_c.float()
                var = x_fp.pow(2).mean(-1, keepdim=True) + eps
                out = (x_fp * torch.rsqrt(var) * w_c.float()).to(x.dtype)
            ctx.save_for_backward(x, weight)
            ctx.eps = eps
            return out

        @staticmethod
        def backward(ctx, grad_output):
            x, weight = ctx.saved_tensors
            eps = ctx.eps
            if backward_fn is not None:
                x_c = x.contiguous()
                w_c = weight.contiguous()
                g_c = grad_output.contiguous()
                N = x_c.shape[0] if x_c.ndim > 1 else 1
                H = x_c.shape[-1]
                grad_x_fp32 = torch.empty(
                    N * H, dtype=torch.float32, device=x.device
                )
                grad_w_fp32 = torch.zeros(H, dtype=torch.float32, device=x.device)
                backward_fn(
                    g_c.data_ptr(), x_c.data_ptr(), w_c.data_ptr(),
                    grad_x_fp32.data_ptr(), grad_w_fp32.data_ptr(),
                    N, H, float(eps)
                )
                torch.cuda.synchronize()
                grad_x = grad_x_fp32.reshape(x_c.shape).to(x.dtype)
                grad_w = grad_w_fp32.to(weight.dtype)
                return grad_x, grad_w, None
            else:
                x_fp = x.float()
                w_fp = weight.float()
                g_fp = grad_output.float()
                rms = torch.sqrt(x_fp.pow(2).mean(-1, keepdim=True) + eps)
                x_norm = x_fp / rms
                grad_w = (g_fp * x_norm).sum(
                    dim=tuple(range(g_fp.ndim - 1))
                ).to(weight.dtype)
                grad_x_norm = g_fp * w_fp
                grad_x = (
                    (grad_x_norm
                     - x_norm * (grad_x_norm * x_norm).mean(-1, keepdim=True))
                    / rms
                ).to(x.dtype)
                return grad_x, grad_w, None

    class RMSNormCustomModule(nn.Module):
        """替换 Qwen3RMSNorm 的自定义 Module"""
        def __init__(self, weight: nn.Parameter, eps: float):
            super().__init__()
            self.weight = weight
            self.variance_epsilon = eps

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            orig_shape = hidden_states.shape
            x_2d = hidden_states.reshape(-1, orig_shape[-1])
            out_2d = RMSNormFunction.apply(
                x_2d, self.weight, self.variance_epsilon
            )
            return out_2d.reshape(orig_shape)

        def extra_repr(self):
            return (
                f"hidden={tuple(self.weight.shape)}, "
                f"eps={self.variance_epsilon}, source=operator_agent"
            )

    return {"RMSNormModule": RMSNormCustomModule}


def _make_gelu_inject_fn(desc: OperatorDesc, so_paths: dict) -> dict:
    """GeLU inject_fn 工厂，与 SiLU 类似。"""
    forward_so = so_paths.get("gelu_forward")
    backward_so = so_paths.get("gelu_backward")

    forward_fn = None
    if forward_so and os.path.exists(forward_so):
        try:
            lib = ctypes.CDLL(forward_so)
            fn = lib.launch_kernel
            fn.restype = None
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            forward_fn = fn
            logger.info("[builtin_ops] ✅ GeLU forward kernel loaded")
        except (OSError, AttributeError) as e:
            logger.warning(f"[builtin_ops] GeLU forward load failed: {e}")

    backward_fn = None
    if backward_so and os.path.exists(backward_so):
        try:
            lib_bwd = ctypes.CDLL(backward_so)
            fn_bwd = lib_bwd.launch_kernel
            fn_bwd.restype = None
            fn_bwd.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_void_p, ctypes.c_int]
            backward_fn = fn_bwd
            logger.info("[builtin_ops] ✅ GeLU backward kernel loaded (float32 grad)")
        except (OSError, AttributeError) as e:
            logger.warning(f"[builtin_ops] GeLU backward load failed: {e}")

    class GeLUCustomFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            x_c = x.contiguous()
            if forward_fn is not None:
                out = torch.empty_like(x_c)
                forward_fn(x_c.data_ptr(), out.data_ptr(), x_c.numel())
                torch.cuda.synchronize()
            else:
                out = F.gelu(x)
            ctx.save_for_backward(x)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            if backward_fn is not None:
                x_c = x.contiguous()
                g_c = grad_output.contiguous()
                grad_in_fp32 = torch.empty(
                    x_c.numel(), dtype=torch.float32, device=x_c.device
                )
                backward_fn(g_c.data_ptr(), x_c.data_ptr(),
                            grad_in_fp32.data_ptr(), x_c.numel())
                torch.cuda.synchronize()
                return grad_in_fp32.reshape(x_c.shape).to(x.dtype)
            else:
                xr = x.float().requires_grad_(True)
                F.gelu(xr).backward(grad_output.float())
                return xr.grad.to(x.dtype)

    def gelu_custom(x):
        return GeLUCustomFunction.apply(x)

    return {"gelu_fn": gelu_custom}


# ═══════════════════════════════════════════════════════════════════
# OperatorDesc 定义
# ═══════════════════════════════════════════════════════════════════

def _rmsnorm_bwd_reference(go, x, weight):
    """RMSNorm backward 的 PyTorch reference 实现。"""
    xr = x.float().requires_grad_(True)
    wr = weight.float().requires_grad_(True)
    out = xr / torch.sqrt(xr.pow(2).mean(-1, keepdim=True) + 1e-6) * wr
    out.backward(go.float())
    return xr.grad.float(), wr.grad.float()


# ── SiLU ───────────────────────────────────────────────────────────

SILU_FORWARD_DESC = OperatorDesc(
    name="silu",
    variant="forward",
    ctypes_argtypes=["void*", "void*", "int"],
    output_arg_indices=[1],
    output_dtypes=["fp16"],
    pytorch_reference=lambda x: F.silu(x),
    input_shapes_fn=lambda tc: {
        "x": torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda")
    },
    scalar_args_fn=lambda tc, inp: [inp["x"].numel()],
    error_threshold=0.05,
    inject_pattern=("attr", "act_fn", "silu"),
    inject_fn=_make_silu_inject_fn,
)

SILU_BACKWARD_DESC = OperatorDesc(
    name="silu",
    variant="backward",
    ctypes_argtypes=["void*", "void*", "void*", "int"],  # go, x, grad_in_fp32, N
    output_arg_indices=[2],
    output_dtypes=["fp32"],   # grad_in 是 float32，防止 fp16 overflow
    pytorch_reference=lambda go, x: (
        go.float() * torch.sigmoid(x.float()) * (1.0 + x.float() * (1.0 - torch.sigmoid(x.float())))
    ),
    input_shapes_fn=lambda tc: {
        "go": torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda"),
        "x":  torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda"),
    },
    scalar_args_fn=lambda tc, inp: [inp["go"].numel()],
    error_threshold=0.05,
    inject_pattern=None,   # backward 由 forward inject_fn 内部处理
    inject_fn=None,
)

# ── RMSNorm ────────────────────────────────────────────────────────

RMSNORM_FORWARD_DESC = OperatorDesc(
    name="rmsnorm",
    variant="forward",
    ctypes_argtypes=["void*", "void*", "void*", "int", "int", "float"],
    output_arg_indices=[2],
    output_dtypes=["fp16"],
    pytorch_reference=lambda x, w: (
        (x.float() / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).half()
    ),
    input_shapes_fn=lambda tc: {
        "x":      torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda"),
        "weight": torch.ones(tc["H"],           dtype=torch.float16, device="cuda"),
    },
    scalar_args_fn=lambda tc, inp: [tc["N"], tc["H"], 1e-6],
    output_shapes_fn=lambda tc, inp, i: (tc["N"], tc["H"]),
    error_threshold=0.05,
    inject_pattern=("module_type", "RMSNorm"),
    inject_fn=_make_rmsnorm_inject_fn,
)

RMSNORM_BACKWARD_DESC = OperatorDesc(
    name="rmsnorm",
    variant="backward",
    ctypes_argtypes=[
        "void*", "void*", "void*",          # go, x, weight
        "void*", "void*",                    # grad_x_fp32, grad_w_fp32
        "int", "int", "float",
    ],
    output_arg_indices=[3, 4],
    output_dtypes=["fp32", "fp32"],          # 两个梯度均用 float32
    pytorch_reference=_rmsnorm_bwd_reference,
    input_shapes_fn=lambda tc: {
        "go":     torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda"),
        "x":      torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda"),
        "weight": torch.ones(tc["H"],           dtype=torch.float16, device="cuda"),
    },
    scalar_args_fn=lambda tc, inp: [tc["N"], tc["H"], 1e-6],
    output_shapes_fn=lambda tc, inp, i: (
        (tc["N"] * tc["H"],) if i == 0 else (tc["H"],)
    ),
    error_threshold=0.05,
    inject_pattern=None,
    inject_fn=None,
)

# ── GeLU ───────────────────────────────────────────────────────────

GELU_FORWARD_DESC = OperatorDesc(
    name="gelu",
    variant="forward",
    ctypes_argtypes=["void*", "void*", "int"],
    output_arg_indices=[1],
    output_dtypes=["fp16"],
    pytorch_reference=lambda x: F.gelu(x),
    input_shapes_fn=lambda tc: {
        "x": torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda")
    },
    scalar_args_fn=lambda tc, inp: [inp["x"].numel()],
    error_threshold=0.05,
    inject_pattern=("attr", "act_fn", "gelu"),
    inject_fn=_make_gelu_inject_fn,
)

GELU_BACKWARD_DESC = OperatorDesc(
    name="gelu",
    variant="backward",
    ctypes_argtypes=["void*", "void*", "void*", "int"],
    output_arg_indices=[2],
    output_dtypes=["fp32"],
    pytorch_reference=lambda go, x: (
        (lambda xr: (F.gelu(xr).backward(go.float()), xr.grad)[1])(
            x.float().requires_grad_(True)
        )
    ),
    input_shapes_fn=lambda tc: {
        "go": torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda"),
        "x":  torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda"),
    },
    scalar_args_fn=lambda tc, inp: [inp["go"].numel()],
    error_threshold=0.05,
    inject_pattern=None,
    inject_fn=None,
)

# ── Softmax ────────────────────────────────────────────────────────
# 仅生成+验证，暂不注入（Qwen3 attention 内部调用，注入复杂度高）

SOFTMAX_FORWARD_DESC = OperatorDesc(
    name="softmax",
    variant="forward",
    ctypes_argtypes=["void*", "void*", "int", "int"],  # x, out, N, C
    output_arg_indices=[1],
    output_dtypes=["fp16"],
    pytorch_reference=lambda x, N, C: F.softmax(x.view(N, C).float(), dim=-1).half().view(N, C),
    input_shapes_fn=lambda tc: {
        "x": torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda"),
    },
    scalar_args_fn=lambda tc, inp: [tc["N"], tc["H"]],
    output_shapes_fn=lambda tc, inp, i: (tc["N"], tc["H"]),
    test_cases=[{"N": 64, "H": 512}, {"N": 32, "H": 1024}],
    error_threshold=0.05,
    inject_pattern=None,
    inject_fn=None,
)

# ── Cross Entropy ──────────────────────────────────────────────────
# 仅生成+验证，暂不注入

def _cross_entropy_reference(logits, targets_int):
    """Cross entropy reference：logits 是 fp16，targets 是 int32 编码为 fp16。"""
    N = logits.shape[0]
    C = logits.shape[1] if logits.ndim > 1 else 1
    targets_long = targets_int.view(-1).long()
    return F.cross_entropy(logits.float().view(N, C), targets_long, reduction="mean")


CROSS_ENTROPY_FORWARD_DESC = OperatorDesc(
    name="cross_entropy",
    variant="forward",
    # 签名：logits(fp16), targets(int32 via void*), loss(fp32 scalar), N, C
    ctypes_argtypes=["void*", "void*", "void*", "int", "int"],
    output_arg_indices=[2],
    output_dtypes=["fp32"],
    pytorch_reference=_cross_entropy_reference,
    input_shapes_fn=lambda tc: {
        "logits":  torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda"),
        "targets": torch.randint(0, tc["H"], (tc["N"],), dtype=torch.int32, device="cuda"),
    },
    scalar_args_fn=lambda tc, inp: [tc["N"], tc["H"]],
    output_shapes_fn=lambda tc, inp, i: (1,),  # scalar loss
    test_cases=[{"N": 64, "H": 512}, {"N": 16, "H": 256}],
    error_threshold=0.05,
    inject_pattern=None,
    inject_fn=None,
)

# ── Embedding ──────────────────────────────────────────────────────
# 仅生成+验证，暂不注入

EMBEDDING_FORWARD_DESC = OperatorDesc(
    name="embedding",
    variant="forward",
    # 签名：weight(fp16), indices(int32), out(fp16), vocab_size, hidden_size
    ctypes_argtypes=["void*", "void*", "void*", "int", "int"],
    output_arg_indices=[2],
    output_dtypes=["fp16"],
    pytorch_reference=lambda w, idx, V, H: F.embedding(idx.long(), w),
    input_shapes_fn=lambda tc: {
        "weight":  torch.randn(tc.get("V", 512), tc["H"], dtype=torch.float16, device="cuda"),
        "indices": torch.randint(0, tc.get("V", 512), (tc["N"],), dtype=torch.int32, device="cuda"),
    },
    scalar_args_fn=lambda tc, inp: [tc.get("V", 512), tc["H"]],
    output_shapes_fn=lambda tc, inp, i: (tc["N"], tc["H"]),
    test_cases=[{"N": 32, "H": 256, "V": 512}],
    error_threshold=0.05,
    inject_pattern=None,  # 暂不注入（embedding 层通常参与梯度，注入需更多工作）
    inject_fn=None,
)


# ═══════════════════════════════════════════════════════════════════
# Linear/GEMM inject_fn（替换 nn.Linear）
# ═══════════════════════════════════════════════════════════════════

def _make_linear_inject_fn(desc: OperatorDesc, so_paths: dict) -> dict:
    """
    Linear inject_fn 工厂：
      给定 matmul_forward.so，创建 CustomLinear Module，
      返回 {"LinearModule": <class>}。

    CustomLinear 保持与 nn.Linear 相同的接口（weight/bias 参数），
    forward 调用 agent 生成的 GEMM kernel，backward 使用 PyTorch autograd。
    ctypes 签名：(A, B, C, M_int, N_int, K_int)
      A = input (M×K), B = weight^T (K×N), C = output (M×N)
    """
    forward_so = so_paths.get("matmul_forward")

    gemm_fn = None
    if forward_so and os.path.exists(forward_so):
        try:
            lib = ctypes.CDLL(forward_so)
            fn = lib.launch_kernel
            fn.restype = None
            fn.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            gemm_fn = fn
            logger.info("[builtin_ops] ✅ Linear/GEMM forward kernel loaded")
        except (OSError, AttributeError) as e:
            logger.warning(f"[builtin_ops] Linear GEMM load failed: {e}, using PyTorch fallback")

    class GEMMFunction(torch.autograd.Function):
        """用 agent GEMM kernel 做 forward，backward 用 PyTorch autograd 自动求导。"""
        @staticmethod
        def forward(ctx, x, weight):
            # x: (M, K), weight: (N, K) → output: (M, N)
            # kernel 需要 B = weight^T → (K, N)，即 weight.t().contiguous()
            x_c = x.contiguous().half()
            w_t = weight.t().contiguous().half()   # (K, N)
            M, K = x_c.shape
            K2, N = w_t.shape
            out = torch.empty(M, N, dtype=torch.float16, device=x.device)
            gemm_fn(x_c.data_ptr(), w_t.data_ptr(), out.data_ptr(), M, N, K)
            torch.cuda.synchronize()
            ctx.save_for_backward(x, weight)
            return out.to(x.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            x, weight = ctx.saved_tensors
            # grad_x = grad_output @ weight，grad_weight = grad_output^T @ x
            grad_x = grad_output.float() @ weight.float()
            grad_weight = grad_output.float().t() @ x.float()
            return grad_x.to(x.dtype), grad_weight.to(weight.dtype)

    class CustomLinear(nn.Module):
        """
        替换 nn.Linear 的自定义模块。
        保留原模块的 weight/bias 参数，forward 用 GEMM kernel（若可用）。
        不改变 in_features/out_features/bias 接口。
        """
        def __init__(self, original_linear: nn.Linear):
            super().__init__()
            self.weight = original_linear.weight        # 复用原始权重 Parameter
            self.bias   = original_linear.bias          # 可以为 None
            self.in_features  = original_linear.in_features
            self.out_features = original_linear.out_features

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            orig_shape = x.shape
            # 展平成 2D：(batch*seq, hidden)
            x_2d = x.reshape(-1, self.in_features)
            if gemm_fn is not None:
                # GEMM kernel 要求 half；先转，算完再转回
                out_2d = GEMMFunction.apply(x_2d.half(), self.weight.half())
                out_2d = out_2d.to(x.dtype)
            else:
                out_2d = F.linear(x_2d, self.weight)
            if self.bias is not None:
                out_2d = out_2d + self.bias
            return out_2d.reshape(*orig_shape[:-1], self.out_features)

        def extra_repr(self) -> str:
            return (
                f"in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias is not None}, "
                f"kernel={'agent' if gemm_fn else 'pytorch'}"
            )

    return {"LinearModule": CustomLinear}


# ── MatMul/Linear OperatorDesc ─────────────────────────────────────

MATMUL_FORWARD_DESC = OperatorDesc(
    name="matmul",
    variant="forward",
    ctypes_argtypes=["void*", "void*", "void*", "int", "int", "int"],
    output_arg_indices=[2],
    output_dtypes=["fp16"],
    pytorch_reference=lambda A, B: torch.matmul(A.float(), B.float()).half(),
    input_shapes_fn=lambda tc: {
        "A": torch.randn(tc.get("M", 64), tc.get("K", 128),
                         dtype=torch.float16, device="cuda"),
        "B": torch.randn(tc.get("K", 128), tc.get("N", 64),
                         dtype=torch.float16, device="cuda"),
    },
    scalar_args_fn=lambda tc, inp: [tc.get("M", 64), tc.get("N", 64), tc.get("K", 128)],
    output_shapes_fn=lambda tc, inp, i: (tc.get("M", 64), tc.get("N", 64)),
    test_cases=[
        {"M": 64,  "K": 4096,  "N": 4096},   # q_proj 尺寸
        {"M": 32,  "K": 4096,  "N": 12288},  # gate_proj 尺寸
    ],
    error_threshold=0.05,
    inject_pattern=("linear_name", ""),   # 匹配所有 nn.Linear（name_substr="" = 全匹配）
    inject_fn=_make_linear_inject_fn,
)


# ═══════════════════════════════════════════════════════════════════
# 注册函数
# ═══════════════════════════════════════════════════════════════════

#: 所有内置算子描述（按名称分组，forward 在前）
ALL_BUILTIN_DESCS: list[OperatorDesc] = [
    SILU_FORWARD_DESC,
    SILU_BACKWARD_DESC,
    RMSNORM_FORWARD_DESC,
    RMSNORM_BACKWARD_DESC,
    GELU_FORWARD_DESC,
    GELU_BACKWARD_DESC,
    MATMUL_FORWARD_DESC,
    SOFTMAX_FORWARD_DESC,
    CROSS_ENTROPY_FORWARD_DESC,
    EMBEDDING_FORWARD_DESC,
]


def register_builtin_ops(registry: "OpRegistry") -> "OpRegistry":
    """
    将所有内置 OperatorDesc 注册进给定的 OpRegistry。
    返回 registry 本身（支持链式调用）。

    用法：
        from operators.builtin_ops import register_builtin_ops
        from operators.op_registry import get_op_registry
        reg = register_builtin_ops(get_op_registry())
    """
    for desc in ALL_BUILTIN_DESCS:
        registry.register(desc)
    logger.info(
        f"[builtin_ops] 注册 {len(ALL_BUILTIN_DESCS)} 个内置算子: "
        f"{[d.key for d in ALL_BUILTIN_DESCS]}"
    )
    return registry
