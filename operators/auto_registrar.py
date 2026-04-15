"""
AutoOpRegistrar — 自动算子识别与注册
=====================================
识别用户任务中需要的、但尚未在 OpRegistry 中注册的算子，
自动生成对应的 OperatorDesc，写入 operators/generated_ops.py，
并立即注册到运行时注册表。

复杂度判断逻辑：
  不再依赖硬编码的算子名集合，而是：
  1. 如果算子在 OPERATOR_TEMPLATES 中 → 读取其 OperatorCategory，按 category 路由
  2. 如果算子不在 OPERATOR_TEMPLATES 中 → 尝试通过名称关键词推断 category
  3. 完全无法推断 → 返回 None（PyTorch fallback）

category → 复杂度映射：
  ELEMENTWISE  → 简单（x, out, N）
  NORMALIZATION → 中等（x, w, out, N, H, eps 或含 bias）
  MATMUL       → 中等（A, B, C, M, N, K）
  REDUCTION    → 中等（x, out, N, C）
  EMBEDDING    → 中等（weight, indices, out, V, H）
  ATTENTION    → 复杂（接口依赖 head_dim/batch，暂不自动推导）
  FUSED        → 复杂（暂不支持）
  COMMUNICATION→ 跳过（分布式通信，不适合 ctypes 注入）

用法：
    from operators.auto_registrar import AutoOpRegistrar
    auto_reg = AutoOpRegistrar()
    missing = auto_reg.find_missing(plan, registry)
    descs = auto_reg.generate_missing_descs(missing)
    auto_reg.write_and_register(descs, registry, "operators/generated_ops.py")
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from operators.op_desc import OperatorDesc

if TYPE_CHECKING:
    from agents.training_analyst import TrainingPlan
    from operators.op_registry import OpRegistry

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# OperatorCategory → 复杂度映射
# ─────────────────────────────────────────────────────────────────
# 直接从 models.operator_ir 导入（唯一来源，不重复定义）
try:
    from models.operator_ir import OperatorCategory
    _HAS_CATEGORY = True
except ImportError:
    _HAS_CATEGORY = False
    OperatorCategory = None  # type: ignore

# category → 处理策略
_CATEGORY_STRATEGY = {}
if _HAS_CATEGORY:
    _CATEGORY_STRATEGY = {
        OperatorCategory.ELEMENTWISE:   "elementwise",   # (x, out, N)
        OperatorCategory.NORMALIZATION: "normalization", # (x, w, out, N, H, eps) or with bias
        OperatorCategory.MATMUL:        "matmul",        # (A, B, C, M, N, K)
        OperatorCategory.REDUCTION:     "reduction",     # (x, out, N, C)
        OperatorCategory.EMBEDDING:     "embedding",     # (weight, indices, out, V, H)
        OperatorCategory.ATTENTION:     "complex",       # 接口复杂，暂不支持
        OperatorCategory.FUSED:         "complex",       # 融合算子，暂不支持
        OperatorCategory.COMMUNICATION: "skip",          # 集合通信，跳过
        OperatorCategory.CONVOLUTION:   "complex",       # 暂不支持
        OperatorCategory.MEMORY:        "complex",       # 暂不支持
    }

# 名称关键词 → category 映射（用于不在 OPERATOR_TEMPLATES 的算子）
_NAME_TO_CATEGORY_HINT = {
    # elementwise
    "relu": "elementwise", "leaky_relu": "elementwise", "elu": "elementwise",
    "tanh": "elementwise", "sigmoid": "elementwise", "hardswish": "elementwise",
    "mish": "elementwise", "selu": "elementwise", "celu": "elementwise",
    "prelu": "elementwise", "swish": "elementwise", "glu": "elementwise",
    # normalization
    "layernorm": "normalization", "rmsnorm": "normalization",
    "batchnorm": "normalization", "groupnorm": "normalization",
    "instancenorm": "normalization", "layer_norm": "normalization",
    "rms_norm": "normalization",
    # matmul
    "matmul": "matmul", "linear": "matmul", "gemm": "matmul",
    "bmm": "matmul", "mm": "matmul",
    # reduction
    "softmax": "reduction", "log_softmax": "reduction",
    "cross_entropy": "reduction", "sum": "reduction",
    "mean": "reduction", "max": "reduction",
    # embedding
    "embedding": "embedding", "embed": "embedding",
    # complex
    "flash_attention": "complex", "attention": "complex",
    "fused_moe": "complex", "moe": "complex",
    # skip
    "allreduce": "skip", "allgather": "skip", "broadcast": "skip",
}

# inject_pattern 按 category + op_name 自动推断
# elementwise: 替换 module.act_fn 属性
# normalization: 替换同类型的 Module
_ELEMENTWISE_ACT_FN_NAMES = {
    "relu", "leaky_relu", "elu", "tanh", "sigmoid",
    "hardswish", "mish", "selu", "celu", "prelu", "swish", "glu",
}

# PyTorch reference lambdas（字符串形式，写入 generated_ops.py 用）
_PYTORCH_REF_STR = {
    "relu":       "lambda x: F.relu(x)",
    "leaky_relu": "lambda x: F.leaky_relu(x, 0.01)",
    "tanh":       "lambda x: torch.tanh(x.float()).half()",
    "sigmoid":    "lambda x: torch.sigmoid(x.float()).half()",
    "elu":        "lambda x: F.elu(x.float()).half()",
    "hardswish":  "lambda x: F.hardswish(x.float()).half()",
    "mish":       "lambda x: F.mish(x.float()).half()",
    "selu":       "lambda x: F.selu(x.float()).half()",
    "celu":       "lambda x: F.celu(x.float()).half()",
    "swish":      "lambda x: F.silu(x)",  # swish = silu
    "glu":        "lambda x: F.glu(x)",
    "layernorm":  "lambda x, w, b: F.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float()).half()",
    "batchnorm":  "lambda x, w, b: F.batch_norm(x.float(), None, None, w.float(), b.float()).half()",
    "groupnorm":  "lambda x, w, b: F.group_norm(x.float(), 1, w.float(), b.float()).half()",
    "matmul":     "lambda A, B: torch.matmul(A.float(), B.float()).half()",
    "linear":     "lambda A, B: torch.matmul(A.float(), B.float()).half()",
    "gemm":       "lambda A, B: torch.matmul(A.float(), B.float()).half()",
    "softmax":    "lambda x, N, C: F.softmax(x.view(N, C).float(), dim=-1).half().view(N, C)",
    "log_softmax":"lambda x, N, C: F.log_softmax(x.view(N, C).float(), dim=-1).half().view(N, C)",
    "embedding":  "lambda w, idx, V, H: F.embedding(idx.long(), w)",
}

# PyTorch reference 可调用对象（运行时用）
_PYTORCH_REF_FN = {
    "relu":       lambda x: F.relu(x),
    "leaky_relu": lambda x: F.leaky_relu(x, 0.01),
    "tanh":       lambda x: torch.tanh(x.float()).half(),
    "sigmoid":    lambda x: torch.sigmoid(x.float()).half(),
    "elu":        lambda x: F.elu(x.float()).half(),
    "hardswish":  lambda x: F.hardswish(x.float()).half(),
    "mish":       lambda x: F.mish(x.float()).half(),
    "selu":       lambda x: F.selu(x.float()).half(),
    "celu":       lambda x: F.celu(x.float()).half(),
    "swish":      lambda x: F.silu(x),
    "glu":        lambda x: F.glu(x),
    "layernorm":  lambda x, w, b: F.layer_norm(
        x.float(), (x.shape[-1],), w.float(), b.float()
    ).half(),
    "matmul":     lambda A, B: torch.matmul(A.float(), B.float()).half(),
    "linear":     lambda A, B: torch.matmul(A.float(), B.float()).half(),
    "gemm":       lambda A, B: torch.matmul(A.float(), B.float()).half(),
    "softmax":    lambda x, N, C: F.softmax(
        x.view(int(N), int(C)).float(), dim=-1
    ).half().view(int(N), int(C)),
    "embedding":  lambda w, idx, V, H: F.embedding(idx.long(), w),
}


# ─────────────────────────────────────────────────────────────────
# AutoOpRegistrar
# ─────────────────────────────────────────────────────────────────

class AutoOpRegistrar:
    """
    自动算子识别 + OperatorDesc 生成 + 写入 generated_ops.py。

    核心改进：不再依赖硬编码的算子名集合，而是通过
    OperatorCategory（来自 OPERATOR_TEMPLATES 或名称推断）
    自动决定 ctypes 接口模板和复杂度分类。
    """

    def __init__(self, output_path: str = "operators/generated_ops.py"):
        self.output_path = output_path
        # 延迟加载 OPERATOR_TEMPLATES（避免循环导入）
        self._templates = None

    @property
    def templates(self) -> dict:
        if self._templates is None:
            try:
                from agents.spec_analyzer import OPERATOR_TEMPLATES
                self._templates = OPERATOR_TEMPLATES
            except ImportError:
                self._templates = {}
        return self._templates

    # ── 1. 推断算子复杂度策略 ──────────────────────────────────────

    def infer_strategy(self, op_name: str) -> str:
        """
        推断算子的处理策略：
          "elementwise"   → ctypes (x, out, N)
          "normalization" → ctypes (x, w, out, N, H, eps) 或含 bias
          "matmul"        → ctypes (A, B, C, M, N, K)
          "reduction"     → ctypes (x, out, N, C)
          "embedding"     → ctypes (weight, indices, out, V, H)
          "complex"       → 返回 None（无法自动推导接口）
          "skip"          → 跳过（通信类算子）
          "unknown"       → 无法识别

        优先级：OPERATOR_TEMPLATES.category > 名称关键词匹配
        """
        # 1. 尝试从 OPERATOR_TEMPLATES 读取 category
        if op_name in self.templates:
            tmpl = self.templates[op_name]
            cat = tmpl.get("category")
            if cat is not None and _HAS_CATEGORY:
                strategy = _CATEGORY_STRATEGY.get(cat, "unknown")
                logger.debug(
                    f"[AutoOpRegistrar] {op_name}: category={cat.value} → {strategy}"
                )
                return strategy

        # 2. 名称关键词匹配（处理不在模板库中的算子）
        op_lower = op_name.lower()
        # 精确匹配
        if op_lower in _NAME_TO_CATEGORY_HINT:
            return _NAME_TO_CATEGORY_HINT[op_lower]
        # 子串匹配
        for keyword, strategy in _NAME_TO_CATEGORY_HINT.items():
            if keyword in op_lower:
                logger.debug(
                    f"[AutoOpRegistrar] {op_name}: 关键词匹配 '{keyword}' → {strategy}"
                )
                return strategy

        return "unknown"

    def explain_complexity(self, op_name: str) -> str:
        """
        返回人类可读的复杂度解释，供日志/调试使用。
        """
        strategy = self.infer_strategy(op_name)
        explanations = {
            "elementwise":   f"{op_name}: 逐元素算子（简单）— 输入1张量→输出1张量，形状不变，ctypes: (x, out, N)",
            "normalization": f"{op_name}: 归一化算子（中等）— 需要 weight/bias 参数，ctypes: (x, w, out, N, H, eps)",
            "matmul":        f"{op_name}: 矩阵乘法（中等）— 两输入一输出，ctypes: (A, B, C, M, N, K)",
            "reduction":     f"{op_name}: 规约算子（中等）— 跨维度计算，ctypes: (x, out, N, C)",
            "embedding":     f"{op_name}: 嵌入查找（中等）— 整数索引→向量，ctypes: (weight, idx, out, V, H)",
            "complex":       f"{op_name}: 复杂算子（无法自动推导）— 接口依赖运行时参数（head_dim/batch等），需手动实现",
            "skip":          f"{op_name}: 通信算子（跳过）— 集合通信不适合 ctypes 单机注入",
            "unknown":       f"{op_name}: 未知算子（不在已知集合中，无法推断接口）",
        }
        return explanations.get(strategy, f"{op_name}: strategy={strategy}")

    # ── 2. 识别缺失算子 ────────────────────────────────────────────

    def find_missing(
        self,
        plan: "TrainingPlan",
        registry: "OpRegistry",
    ) -> list[str]:
        """
        返回 plan.all_operators() 中、系统能处理但 registry 尚未注册的算子名。
        自动跳过 strategy in ("skip", "complex") 的算子。
        """
        registered = set(registry.names())
        needed = set(plan.all_operators())
        missing = []
        for op in sorted(needed):
            if op in registered:
                continue
            strategy = self.infer_strategy(op)
            if strategy in ("skip",):
                logger.debug(f"[AutoOpRegistrar] {op}: strategy=skip，跳过")
                continue
            if strategy == "unknown":
                logger.debug(f"[AutoOpRegistrar] {op}: strategy=unknown，跳过")
                continue
            # complex 算子也加入 missing 列表（会在 generate 时返回 None + 打印提示）
            missing.append(op)
        if missing:
            logger.info(f"[AutoOpRegistrar] 发现未注册算子: {missing}")
        return missing

    # ── 3. 生成单个 OperatorDesc ───────────────────────────────────

    def generate_op_desc(self, op_name: str) -> Optional[OperatorDesc]:
        """
        为单个算子名生成 OperatorDesc。
        基于 infer_strategy() 的结果路由到对应模板构造器。
        """
        strategy = self.infer_strategy(op_name)
        logger.info(
            f"[AutoOpRegistrar] {op_name}: {self.explain_complexity(op_name)}"
        )

        if strategy == "elementwise":
            return self._make_elementwise_desc(op_name)
        elif strategy == "normalization":
            return self._make_normalization_desc(op_name)
        elif strategy == "matmul":
            return self._make_matmul_desc(op_name)
        elif strategy == "reduction":
            return self._make_reduction_desc(op_name)
        elif strategy == "embedding":
            return self._make_embedding_desc(op_name)
        elif strategy == "complex":
            logger.warning(
                f"[AutoOpRegistrar] {op_name}: 复杂算子，ctypes接口需手动实现，"
                f"暂用 PyTorch fallback"
            )
            return None
        else:
            logger.warning(
                f"[AutoOpRegistrar] {op_name}: 无法自动推导接口（strategy={strategy}），"
                f"使用 PyTorch fallback"
            )
            return None

    def generate_missing_descs(self, missing_ops: list[str]) -> list[OperatorDesc]:
        """批量生成，自动过滤 None，并打印每个算子的复杂度说明。"""
        descs = []
        print(f"\n[AutoOpRegistrar] 算子复杂度分析:")
        for op in missing_ops:
            print(f"  {self.explain_complexity(op)}")
            d = self.generate_op_desc(op)
            if d:
                descs.append(d)
                logger.info(f"[AutoOpRegistrar] ✅ 已生成 OperatorDesc: {d.key}")
            else:
                logger.warning(f"[AutoOpRegistrar] ⚠ {op}: 无法生成描述，PyTorch fallback")
        return descs

    # ── 4. 写入 generated_ops.py + 注册 ───────────────────────────

    def write_and_register(
        self,
        descs: list[OperatorDesc],
        registry: "OpRegistry",
        path: Optional[str] = None,
    ) -> None:
        """将 descs 写入 generated_ops.py（幂等），并注册到 registry。"""
        if not descs:
            return
        target = path or self.output_path
        self._write_generated_ops(descs, target)
        for desc in descs:
            registry.register(desc)
        logger.info(
            f"[AutoOpRegistrar] 已写入 {target} 并注册 {len(descs)} 个算子: "
            f"{[d.key for d in descs]}"
        )

    # ── 内部：各策略 OperatorDesc 构造器 ──────────────────────────

    def _make_elementwise_desc(self, op_name: str) -> OperatorDesc:
        """elementwise: (x_half, out_half, N_int)"""
        ref_fn = _PYTORCH_REF_FN.get(op_name)
        # inject_pattern: 按 act_fn 属性替换
        inject = (("attr", "act_fn", op_name)
                  if op_name in _ELEMENTWISE_ACT_FN_NAMES else None)
        return OperatorDesc(
            name=op_name,
            variant="forward",
            ctypes_argtypes=["void*", "void*", "int"],
            output_arg_indices=[1],
            output_dtypes=["fp16"],
            pytorch_reference=ref_fn,
            input_shapes_fn=lambda tc: {
                "x": torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda")
            },
            scalar_args_fn=lambda tc, inp: [inp["x"].numel()],
            error_threshold=0.05,
            inject_pattern=inject,
            inject_fn=None,
        )

    def _make_normalization_desc(self, op_name: str) -> OperatorDesc:
        """
        归一化算子：根据算子名判断是否有 bias。
        - layernorm / batchnorm / groupnorm → 有 bias：(x, w, bias, out, N, H, eps)
        - rmsnorm 类似 → 无 bias：(x, w, out, N, H, eps)
        """
        has_bias = any(k in op_name.lower() for k in ("layer", "batch", "group", "instance"))

        # inject_pattern：按 module 类型名匹配
        type_substr = {
            "layernorm": "LayerNorm", "layer_norm": "LayerNorm",
            "batchnorm": "BatchNorm", "groupnorm": "GroupNorm",
            "instancenorm": "InstanceNorm",
            "rmsnorm": "RMSNorm", "rms_norm": "RMSNorm",
        }.get(op_name.lower(), op_name.capitalize())
        inject = ("module_type", type_substr)

        if has_bias:
            ref_fn = _PYTORCH_REF_FN.get(op_name)
            return OperatorDesc(
                name=op_name,
                variant="forward",
                ctypes_argtypes=["void*", "void*", "void*", "void*", "int", "int", "float"],
                output_arg_indices=[3],
                output_dtypes=["fp16"],
                pytorch_reference=ref_fn,
                input_shapes_fn=lambda tc: {
                    "x":      torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda"),
                    "weight": torch.ones(tc["H"],           dtype=torch.float16, device="cuda"),
                    "bias":   torch.zeros(tc["H"],          dtype=torch.float16, device="cuda"),
                },
                scalar_args_fn=lambda tc, inp: [tc["N"], tc["H"], 1e-5],
                output_shapes_fn=lambda tc, inp, i: (tc["N"], tc["H"]),
                error_threshold=0.05,
                inject_pattern=inject,
                inject_fn=None,
            )
        else:
            # RMSNorm 风格（无 bias）
            return OperatorDesc(
                name=op_name,
                variant="forward",
                ctypes_argtypes=["void*", "void*", "void*", "int", "int", "float"],
                output_arg_indices=[2],
                output_dtypes=["fp16"],
                pytorch_reference=lambda x, w: (
                    x.float() / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
                    * w.float()
                ).half(),
                input_shapes_fn=lambda tc: {
                    "x":      torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda"),
                    "weight": torch.ones(tc["H"],           dtype=torch.float16, device="cuda"),
                },
                scalar_args_fn=lambda tc, inp: [tc["N"], tc["H"], 1e-6],
                output_shapes_fn=lambda tc, inp, i: (tc["N"], tc["H"]),
                error_threshold=0.05,
                inject_pattern=inject,
                inject_fn=None,
            )

    def _make_matmul_desc(self, op_name: str) -> OperatorDesc:
        """matmul: (A_half, B_half, C_half, M_int, N_int, K_int)"""
        return OperatorDesc(
            name=op_name,
            variant="forward",
            ctypes_argtypes=["void*", "void*", "void*", "int", "int", "int"],
            output_arg_indices=[2],
            output_dtypes=["fp16"],
            pytorch_reference=_PYTORCH_REF_FN.get(op_name, lambda A, B: torch.matmul(A.float(), B.float()).half()),
            input_shapes_fn=lambda tc: {
                "A": torch.randn(tc.get("M", 64), tc.get("K", 128),
                                 dtype=torch.float16, device="cuda"),
                "B": torch.randn(tc.get("K", 128), tc.get("N", 64),
                                 dtype=torch.float16, device="cuda"),
            },
            scalar_args_fn=lambda tc, inp: [tc.get("M", 64), tc.get("N", 64), tc.get("K", 128)],
            output_shapes_fn=lambda tc, inp, i: (tc.get("M", 64), tc.get("N", 64)),
            test_cases=[{"M": 64, "K": 128, "N": 64}, {"M": 32, "K": 256, "N": 32}],
            error_threshold=0.05,
            inject_pattern=None,  # Linear 注入复杂，暂不自动注入
            inject_fn=None,
        )

    def _make_reduction_desc(self, op_name: str) -> OperatorDesc:
        """reduction (如 softmax): (x_half, out_half, N_int, C_int)"""
        ref_fn = _PYTORCH_REF_FN.get(op_name)
        return OperatorDesc(
            name=op_name,
            variant="forward",
            ctypes_argtypes=["void*", "void*", "int", "int"],
            output_arg_indices=[1],
            output_dtypes=["fp16"],
            pytorch_reference=ref_fn,
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

    def _make_embedding_desc(self, op_name: str) -> OperatorDesc:
        """embedding: (weight_half, indices_int32, out_half, vocab_size_int, hidden_int)"""
        return OperatorDesc(
            name=op_name,
            variant="forward",
            ctypes_argtypes=["void*", "void*", "void*", "int", "int"],
            output_arg_indices=[2],
            output_dtypes=["fp16"],
            pytorch_reference=lambda w, idx, V, H: F.embedding(idx.long(), w),
            input_shapes_fn=lambda tc: {
                "weight":  torch.randn(tc.get("V", 512), tc["H"],
                                       dtype=torch.float16, device="cuda"),
                "indices": torch.randint(0, tc.get("V", 512), (tc["N"],),
                                         dtype=torch.int32, device="cuda"),
            },
            scalar_args_fn=lambda tc, inp: [tc.get("V", 512), tc["H"]],
            output_shapes_fn=lambda tc, inp, i: (tc["N"], tc["H"]),
            test_cases=[{"N": 32, "H": 256, "V": 512}],
            error_threshold=0.05,
            inject_pattern=None,
            inject_fn=None,
        )

    # ── 内部：写入 generated_ops.py ────────────────────────────────

    def _read_existing_keys(self, path: str) -> set[str]:
        if not os.path.exists(path):
            return set()
        try:
            with open(path) as f:
                content = f.read()
            keys = set()
            for m in re.finditer(
                r"OperatorDesc\([^)]*name=[\"'](\w+)[\"'][^)]*variant=[\"'](\w+)[\"']",
                content, re.DOTALL
            ):
                keys.add(f"{m.group(1)}_{m.group(2)}")
            return keys
        except Exception as e:
            logger.warning(f"[AutoOpRegistrar] 读取已有 keys 失败: {e}")
            return set()

    def _strategy_for_python(self, desc: OperatorDesc) -> str:
        """用于生成 Python 文件中的注释，说明算子策略。"""
        return self.infer_strategy(desc.name)

    def _desc_to_python(self, desc: OperatorDesc) -> str:
        """将 OperatorDesc 序列化为 Python 源码字符串。"""
        op = desc.name
        var = desc.variant
        var_name = f"{op.upper()}_{var.upper()}_DESC"
        strategy = self._strategy_for_python(desc)

        args_str = str(desc.ctypes_argtypes)
        out_idx_str = str(desc.output_arg_indices)
        out_dtype_str = str(desc.output_dtypes)
        ref_str = _PYTORCH_REF_STR.get(op, "None  # TODO: 填写 PyTorch reference")

        # input_shapes_fn / scalar_args_fn 按 strategy 生成
        if strategy == "elementwise":
            shapes_str = (
                'lambda tc: {"x": torch.randn(tc["N"], tc["H"], '
                'dtype=torch.float16, device="cuda")}'
            )
            scalar_str = 'lambda tc, inp: [inp["x"].numel()]'
        elif strategy == "normalization":
            has_bias = len(desc.ctypes_argtypes) >= 7 and desc.ctypes_argtypes.count("void*") >= 4
            if has_bias:
                shapes_str = (
                    'lambda tc: {\n'
                    '        "x":      torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda"),\n'
                    '        "weight": torch.ones(tc["H"],           dtype=torch.float16, device="cuda"),\n'
                    '        "bias":   torch.zeros(tc["H"],          dtype=torch.float16, device="cuda"),\n'
                    '    }'
                )
                scalar_str = 'lambda tc, inp: [tc["N"], tc["H"], 1e-5]'
            else:
                shapes_str = (
                    'lambda tc: {\n'
                    '        "x":      torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda"),\n'
                    '        "weight": torch.ones(tc["H"],           dtype=torch.float16, device="cuda"),\n'
                    '    }'
                )
                scalar_str = 'lambda tc, inp: [tc["N"], tc["H"], 1e-6]'
        elif strategy == "matmul":
            shapes_str = (
                'lambda tc: {\n'
                '        "A": torch.randn(tc.get("M", 64), tc.get("K", 128), dtype=torch.float16, device="cuda"),\n'
                '        "B": torch.randn(tc.get("K", 128), tc.get("N", 64), dtype=torch.float16, device="cuda"),\n'
                '    }'
            )
            scalar_str = 'lambda tc, inp: [tc.get("M", 64), tc.get("N", 64), tc.get("K", 128)]'
        elif strategy == "reduction":
            shapes_str = (
                'lambda tc: {"x": torch.randn(tc["N"], tc["H"], dtype=torch.float16, device="cuda")}'
            )
            scalar_str = 'lambda tc, inp: [tc["N"], tc["H"]]'
        elif strategy == "embedding":
            shapes_str = (
                'lambda tc: {\n'
                '        "weight":  torch.randn(tc.get("V", 512), tc["H"], dtype=torch.float16, device="cuda"),\n'
                '        "indices": torch.randint(0, tc.get("V", 512), (tc["N"],), dtype=torch.int32, device="cuda"),\n'
                '    }'
            )
            scalar_str = 'lambda tc, inp: [tc.get("V", 512), tc["H"]]'
        else:
            shapes_str = "None  # TODO: 填写 input_shapes_fn"
            scalar_str = "None  # TODO: 填写 scalar_args_fn"

        inject_str = repr(desc.inject_pattern) if desc.inject_pattern else "None"
        test_cases_str = repr(desc.test_cases) if desc.test_cases else "[]"

        return (
            f"# strategy: {strategy}\n"
            f"{var_name} = OperatorDesc(\n"
            f"    name={op!r},\n"
            f"    variant={var!r},\n"
            f"    ctypes_argtypes={args_str},\n"
            f"    output_arg_indices={out_idx_str},\n"
            f"    output_dtypes={out_dtype_str},\n"
            f"    pytorch_reference={ref_str},\n"
            f"    input_shapes_fn={shapes_str},\n"
            f"    scalar_args_fn={scalar_str},\n"
            f"    test_cases={test_cases_str},\n"
            f"    error_threshold=0.05,\n"
            f"    inject_pattern={inject_str},\n"
            f"    inject_fn=None,   # 如需注入模型，请手动实现此工厂函数\n"
            f")"
        )

    def _write_generated_ops(self, descs: list[OperatorDesc], path: str) -> None:
        existing_keys = self._read_existing_keys(path)
        new_descs = [d for d in descs if d.key not in existing_keys]
        if not new_descs:
            logger.info(f"[AutoOpRegistrar] {path} 中所有算子已存在，无需写入")
            return

        is_new_file = not os.path.exists(path) or os.path.getsize(path) == 0
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        with open(path, "a") as f:
            if is_new_file:
                f.write(
                    '"""\noperators/generated_ops.py\n'
                    '==========================\n'
                    '自动生成的算子描述，由 AutoOpRegistrar 写入。\n'
                    '可手动编辑（如填写 inject_fn），系统下次运行时自动加载。\n'
                    '"""\nfrom __future__ import annotations\n'
                    'import torch\nimport torch.nn.functional as F\n'
                    'from operators.op_desc import OperatorDesc\n\n'
                )

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            f.write(f"\n# ── Auto-generated {timestamp} ──────────────────────────────────────\n\n")

            reg_names = []
            for desc in new_descs:
                f.write(self._desc_to_python(desc))
                f.write("\n\n")
                reg_names.append(f"{desc.name.upper()}_{desc.variant.upper()}_DESC")

            func_ts = timestamp.replace("-", "").replace(" ", "_").replace(":", "")
            f.write(f"\ndef register_generated_ops_{func_ts}(registry):\n")
            f.write(f'    """注册本批次自动生成的算子（{timestamp}）"""\n')
            for vname in reg_names:
                f.write(f"    registry.register({vname})\n")
            f.write("\n")

        logger.info(
            f"[AutoOpRegistrar] 已写入 {len(new_descs)} 个算子到 {path}: "
            f"{[d.key for d in new_descs]}"
        )

