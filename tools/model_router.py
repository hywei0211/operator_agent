"""
多模型路由器
===========
根据算子复杂度将 LLM 调用路由到不同模型：
- 简单算子（gelu, silu, relu）→ 快速模型
- 复杂算子（flash_attention, matmul）→ 强模型
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ModelRouter:
    """根据任务复杂度路由到不同 LLM 客户端"""

    # 简单算子：纯逐元素操作，结构简单
    SIMPLE_OPS = {
        "gelu", "silu", "relu", "sigmoid", "tanh",
        "softmax", "layernorm", "rmsnorm", "embedding",
        "dropout", "add", "multiply",
    }

    # 复杂算子：需要 tiling、共享内存、多级归约
    COMPLEX_OPS = {
        "flash_attention", "matmul", "fused_moe",
        "conv2d", "grouped_matmul", "multi_head_attention",
    }

    def __init__(self, fast_client, strong_client):
        """
        Args:
            fast_client: 快速/便宜的 LLM 客户端（如 qwen-plus）
            strong_client: 强大的 LLM 客户端（如 qwen3-235b）
        """
        self.fast = fast_client
        self.strong = strong_client

    def select_client(
        self,
        operator_name: str = "",
        task_type: str = "codegen",
    ):
        """根据算子名和任务类型选择 LLM 客户端"""
        op = operator_name.lower().replace("-", "_").replace(" ", "_")

        # 非代码生成任务（如 spec 分析）用快速模型
        if task_type in ("spec_analysis", "hardware_profiling"):
            logger.debug(f"[ModelRouter] {op} ({task_type}) -> fast")
            return self.fast

        # 简单算子用快速模型
        if op in self.SIMPLE_OPS:
            logger.debug(f"[ModelRouter] {op} -> fast (simple op)")
            return self.fast

        # 复杂算子用强模型
        logger.debug(f"[ModelRouter] {op} -> strong")
        return self.strong
