"""
operators/generated_ops.py
==========================
自动生成的算子描述，由 AutoOpRegistrar 写入。
可以手动编辑（如填写 inject_fn），系统下次运行时会自动加载。
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from operators.op_desc import OperatorDesc

# ── Auto-generated 2026-04-14 17:11 ──────────────────────────────────────

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
    test_cases=[{"M": 64, "K": 128, "N": 64}, {"M": 32, "K": 256, "N": 32}],
    error_threshold=0.05,
    inject_pattern=None,
    inject_fn=None,   # 如需注入模型，请手动实现此工厂函数
)



def register_generated_ops_20260414_1711(registry):
    """注册本批次自动生成的算子（2026-04-14 17:11）"""
    registry.register(MATMUL_FORWARD_DESC)

