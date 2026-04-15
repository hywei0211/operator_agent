"""
OperatorDesc — 算子描述类
==========================
一个算子 variant（forward 或 backward）的完整描述。
注册进 OpRegistry 后，系统可自动完成：
  生成 → 编译 retry → ctypes 验证 → 模型注入

用法示例：
    desc = OperatorDesc(
        name="silu", variant="forward",
        ctypes_argtypes=["void*", "void*", "int"],
        output_arg_indices=[1],
        output_dtypes=["fp16"],
        pytorch_reference=lambda x: F.silu(x),
        input_shapes_fn=lambda tc: {"x": torch.randn(tc["N"], tc["H"],
                                    dtype=torch.float16, device="cuda")},
        scalar_args_fn=lambda tc, inp: [inp["x"].numel()],
        inject_pattern=("attr", "act_fn", "silu"),
        inject_fn=my_inject_factory,
    )
"""
from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import torch

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# ctypes 类型字符串 → ctypes 对象映射
# ─────────────────────────────────────────────────────────────────

_CTYPE_MAP: dict[str, type] = {
    "void*":   ctypes.c_void_p,
    "void *":  ctypes.c_void_p,
    "float*":  ctypes.c_void_p,   # GPU 指针，统一用 c_void_p
    "half*":   ctypes.c_void_p,
    "int*":    ctypes.c_void_p,
    "int":     ctypes.c_int,
    "int32":   ctypes.c_int,
    "int32_t": ctypes.c_int,
    "int64":   ctypes.c_long,
    "int64_t": ctypes.c_long,
    "float":   ctypes.c_float,
    "double":  ctypes.c_double,
    "bool":    ctypes.c_bool,
}

_POINTER_TYPES = frozenset({"void*", "void *", "float*", "half*", "int*"})

# dtype 字符串 → torch.dtype
_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp16":  torch.float16,
    "bf16":  torch.bfloat16,
    "fp32":  torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
}

# ─────────────────────────────────────────────────────────────────
# inject_pattern 类型定义
# ─────────────────────────────────────────────────────────────────

# inject_pattern 支持四种格式：
#   ("attr",        attr_name,   type_substr)  → 匹配 module.<attr_name>，type(attr).__name__ 含 type_substr
#   ("module_type", type_substr)               → 匹配 type(module).__name__ 含 type_substr
#   ("linear_name", name_substr)               → 匹配 named_children 中名字含 name_substr 的 Linear
#   callable(name: str, module) -> bool        → 自定义匹配
InjectPattern = Optional[Union[tuple, Callable]]


# ─────────────────────────────────────────────────────────────────
# 核心 dataclass
# ─────────────────────────────────────────────────────────────────

@dataclass
class OperatorDesc:
    """
    单个算子 variant 的完整描述。

    字段分为四组：
      A. 身份信息
      B. ctypes 接口规格（verify 用）
      C. PyTorch 参考实现（verify 对比用）
      D. 模型注入规格（patch_model 用）
    """

    # ── A. 身份 ───────────────────────────────────────────────────
    name: str
    """算子名称，如 "silu"、"softmax"。"""

    variant: str = "forward"
    """
    "forward" / "backward"。
    key = f"{name}_{variant}"，用于 OpRegistry 索引。
    """

    # ── B. ctypes 接口规格 ────────────────────────────────────────
    ctypes_argtypes: list[str] = field(default_factory=list)
    """
    ctypes 参数类型列表，对应 launch_kernel C 函数签名。
    使用字符串，verify_kernel 会自动转换为 ctypes 对象。

    示例（silu forward）：
        ["void*", "void*", "int"]
        void launch_kernel(half* x, half* out, int numel)

    示例（rmsnorm backward）：
        ["void*", "void*", "void*", "void*", "void*", "int", "int", "float"]
        void launch_kernel(half* go, half* x, half* w,
                           float* gx_fp32, float* gw_fp32,
                           int N, int H, float eps)
    """

    output_arg_indices: list[int] = field(default_factory=list)
    """
    ctypes_argtypes 中，哪几个位置是输出指针。

    示例（silu forward，[x, out, numel]）：output_arg_indices = [1]
    示例（rmsnorm backward，[go, x, w, gx_fp32, gw_fp32, N, H, eps]）：output_arg_indices = [3, 4]
    """

    output_dtypes: list[str] = field(default_factory=list)
    """
    每个输出的数据精度，与 output_arg_indices 一一对应。
    可选值：fp16 / bf16 / fp32 / int32 / int64

    backward kernel 的梯度输出通常用 "fp32"（防止 fp16 overflow → NaN）。
    若为空列表，自动用 "fp16" 填充。
    """

    # ── C. PyTorch 参考实现（验证用）─────────────────────────────
    pytorch_reference: Optional[Callable] = None
    """
    签名：fn(*inputs) -> Tensor | tuple[Tensor, ...]

    输入列表与 input_shapes_fn 返回的字典值（按序）对应。
    若为 None，验证时只做 NaN 检查，视为通过。

    示例（silu forward）：
        lambda x: F.silu(x)

    示例（rmsnorm backward，返回两个梯度）：
        lambda go, x, w: rmsnorm_bwd_ref(go, x, w)   # 返回 (grad_x, grad_w)
    """

    input_shapes_fn: Optional[Callable] = None
    """
    签名：fn(test_case: dict) -> dict[str, Tensor]

    给定一组测试参数（N, H 等），生成所有输入张量。
    返回字典：key 仅用于调试，value 按顺序对应 ctypes 签名中【非输出】的指针参数。

    示例（silu forward）：
        lambda tc: {"x": torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda")}

    示例（rmsnorm backward）：
        lambda tc: {
            "go":     torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda"),
            "x":      torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda"),
            "weight": torch.ones(tc["H"],           dtype=torch.float16, device="cuda"),
        }
    """

    scalar_args_fn: Optional[Callable] = None
    """
    签名：fn(test_case: dict, input_tensors: dict) -> list

    返回 ctypes 签名中所有非指针参数的值列表（int/float），
    按 ctypes_argtypes 中非 void*/float*/half* 参数的顺序。

    示例（silu forward，签名 [void*, void*, int]）：
        lambda tc, inp: [inp["x"].numel()]

    示例（rmsnorm backward，签名中 [..., int, int, float]）：
        lambda tc, inp: [tc["N"], tc["H"], 1e-6]
    """

    output_shapes_fn: Optional[Callable] = None
    """
    （可选）自定义输出 buffer 的 shape。
    签名：fn(test_case: dict, input_tensors: dict, out_idx: int) -> tuple[int, ...]

    若为 None，输出 buffer shape 默认为 (first_input.numel(),)（flat 一维）。
    当输出形状与第一个输入 numel 不同时（如 cross_entropy loss 是标量）需要指定。
    """

    test_cases: list[dict] = field(default_factory=list)
    """
    验证测试用例列表，每个 dict 包含 input_shapes_fn/scalar_args_fn 需要的参数。
    若为空，自动使用 default_test_cases()。
    """

    error_threshold: float = 0.05
    """相对误差阈值，低于此值视为验证通过。"""

    # ── D. 模型注入规格 ───────────────────────────────────────────
    inject_pattern: InjectPattern = None
    """
    在模型中找到注入点的模式。

    四种格式：
        ("attr", attr_name, type_substr)  → 替换属性（如 act_fn）
        ("module_type", type_substr)      → 按类型名替换整个子模块
        ("linear_name", name_substr)      → 按名字替换 nn.Linear（name_substr="" 匹配所有）
        callable(name, module) -> bool    → 自定义匹配

    为 None 时不注入（如 backward、softmax 等暂不替换的算子）。
    """

    inject_fn: Optional[Callable] = None
    """
    工厂函数，负责创建注入模型的对象。
    签名：fn(desc: OperatorDesc, so_paths: dict[str, str | None]) -> dict[str, object]

    so_paths 是 {kernel_key: so_path_or_None} 字典，其中包含：
        f"{desc.name}_forward"  → forward kernel so 路径（可能为 None）
        f"{desc.name}_backward" → backward kernel so 路径（可能为 None）

    返回字典：将被并入 custom_fn_map，供 patch_model 使用。
    通常返回类似 {"silu_fn": wrapper_fn} 或 {"RMSNormModule": SomeClass}。

    为 None 时（如 backward variant）不直接注入。
    """

    # ─────────────────────────────────────────────────────────────
    # 属性 & 工具方法
    # ─────────────────────────────────────────────────────────────

    @property
    def key(self) -> str:
        """注册表索引键，格式为 '{name}_{variant}'。"""
        return f"{self.name}_{self.variant}"

    def resolved_ctypes_argtypes(self) -> list:
        """
        将字符串类型列表转为 ctypes 类型对象列表。
        供 ctypes 函数 fn.argtypes = ... 使用。
        """
        result = []
        for t in self.ctypes_argtypes:
            ct = _CTYPE_MAP.get(t.strip().lower()) or _CTYPE_MAP.get(t.strip())
            if ct is None:
                raise ValueError(
                    f"[OperatorDesc({self.key})] 未知 ctypes 类型: {t!r}。"
                    f"支持: {list(_CTYPE_MAP.keys())}"
                )
            result.append(ct)
        return result

    def resolved_output_dtypes(self) -> list[str]:
        """
        返回每个输出的 dtype 字符串列表。
        若 output_dtypes 为空，自动用 'fp16' 填充（与 output_arg_indices 等长）。
        """
        if not self.output_dtypes:
            return ["fp16"] * len(self.output_arg_indices)
        return list(self.output_dtypes)

    def to_torch_dtype(self, dtype_str: str) -> torch.dtype:
        """将 dtype 字符串转为 torch.dtype。"""
        dtype = _DTYPE_MAP.get(dtype_str.strip().lower())
        if dtype is None:
            raise ValueError(
                f"[OperatorDesc({self.key})] 未知 dtype: {dtype_str!r}。"
                f"支持: {list(_DTYPE_MAP.keys())}"
            )
        return dtype

    def is_pointer_type(self, type_str: str) -> bool:
        """判断 ctypes 类型字符串是否是指针（GPU 张量）。"""
        return type_str.strip().lower() in _POINTER_TYPES

    def default_test_cases(self) -> list[dict]:
        """当 test_cases 为空时使用的默认测试组合。"""
        return [
            {"N": 64,  "H": 1024},
            {"N": 8,   "H": 3072},
            {"N": 16,  "H": 1024},
        ]

    def __repr__(self) -> str:
        return (
            f"OperatorDesc(key={self.key!r}, "
            f"ctypes={self.ctypes_argtypes}, "
            f"out_idx={self.output_arg_indices}, "
            f"out_dtypes={self.output_dtypes}, "
            f"inject={self.inject_pattern})"
        )
