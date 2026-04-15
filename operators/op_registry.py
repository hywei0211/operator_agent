"""
OpRegistry — 算子注册中心
==========================
使用方式：
    from operators.op_registry import get_op_registry
    reg = get_op_registry()
    reg.register(my_desc)
    fn_map = reg.build_custom_fn_map(so_paths)
"""
from __future__ import annotations

import logging
from typing import Iterator, Optional

from operators.op_desc import OperatorDesc

logger = logging.getLogger(__name__)


class OpRegistry:
    """
    算子描述注册中心。

    设计原则：
      - 以 desc.key（= "{name}_{variant}"）为索引
      - 同一 key 重复注册时覆盖（方便热更新）
      - build_custom_fn_map 负责将 desc + so_paths 翻译为注入函数
    """

    def __init__(self):
        # key → OperatorDesc，保持插入顺序
        self._descs: dict[str, OperatorDesc] = {}

    # ─────────────────────────────────────────────────────────────
    # 注册 & 查找
    # ─────────────────────────────────────────────────────────────

    def register(self, desc: OperatorDesc) -> "OpRegistry":
        """
        注册一个 OperatorDesc。
        返回 self，支持链式调用：
            reg.register(silu_fwd).register(silu_bwd)
        """
        if desc.key in self._descs:
            logger.warning(f"[OpRegistry] 覆盖已有算子: {desc.key}")
        self._descs[desc.key] = desc
        logger.debug(f"[OpRegistry] 注册算子: {desc.key}")
        return self

    def unregister(self, key: str) -> None:
        """移除一个算子描述（测试用）。"""
        self._descs.pop(key, None)

    def get(self, key: str) -> Optional[OperatorDesc]:
        """按 key 查找，不存在返回 None。"""
        return self._descs.get(key)

    def get_by_name(self, name: str) -> list[OperatorDesc]:
        """返回同名的所有 variant。"""
        return [d for d in self._descs.values() if d.name == name]

    def get_all(self) -> list[OperatorDesc]:
        """返回所有已注册算子（按注册顺序）。"""
        return list(self._descs.values())

    def get_forward_descs(self) -> list[OperatorDesc]:
        """返回所有 forward variant。"""
        return [d for d in self._descs.values() if d.variant == "forward"]

    def get_backward_descs(self) -> list[OperatorDesc]:
        """返回所有 backward variant。"""
        return [d for d in self._descs.values() if d.variant == "backward"]

    def names(self) -> list[str]:
        """返回所有不重复的算子名（按字母顺序）。"""
        return sorted(set(d.name for d in self._descs.values()))

    # ─────────────────────────────────────────────────────────────
    # 核心方法：构建注入函数映射
    # ─────────────────────────────────────────────────────────────

    def build_custom_fn_map(
        self,
        so_paths: dict[str, Optional[str]],
    ) -> dict[str, object]:
        """
        给定编译好的 so_paths（key = "{name}_{variant}"），
        调用每个 forward desc 的 inject_fn 工厂，返回可注入模型的函数/类映射。

        - so_path 为 None → inject_fn 接收 None，由工厂自行处理 PyTorch fallback
        - inject_fn 为 None 的 desc 跳过

        返回示例：
            {
                "silu_fn":       <SiLUCustomFunction wrapper>,
                "RMSNormModule": <class RMSNormCustomModule>,
            }
        """
        fn_map: dict[str, object] = {}
        processed_names: set[str] = set()

        for desc in self.get_forward_descs():
            if desc.inject_fn is None:
                continue
            if desc.name in processed_names:
                continue
            processed_names.add(desc.name)

            try:
                result = desc.inject_fn(desc, so_paths)
                if isinstance(result, dict):
                    fn_map.update(result)
                    logger.info(
                        f"[OpRegistry] build_fn_map: {desc.name} → keys={list(result.keys())}"
                    )
                else:
                    # 单返回值：以 f"{name}_fn" 为 key
                    fn_map[f"{desc.name}_fn"] = result
                    logger.info(
                        f"[OpRegistry] build_fn_map: {desc.name} → {desc.name}_fn"
                    )
            except Exception as e:
                logger.error(
                    f"[OpRegistry] {desc.key} inject_fn 失败: {e}，跳过"
                )

        return fn_map

    # ─────────────────────────────────────────────────────────────
    # 容器协议
    # ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._descs)

    def __iter__(self) -> Iterator[OperatorDesc]:
        return iter(self._descs.values())

    def __contains__(self, key: str) -> bool:
        return key in self._descs

    # ─────────────────────────────────────────────────────────────
    # 调试
    # ─────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """返回多行字符串，显示所有已注册算子及其 variant。"""
        if not self._descs:
            return "OpRegistry (empty)"
        lines = [f"OpRegistry ({len(self)} entries, {len(self.names())} ops):"]
        for name in self.names():
            variants = [d.variant for d in self.get_by_name(name)]
            inject = [
                d.inject_pattern is not None
                for d in self.get_by_name(name)
                if d.variant == "forward"
            ]
            inject_str = "✅ injectable" if any(inject) else "— gen/verify only"
            lines.append(f"  {name:20s} [{', '.join(variants)}]  {inject_str}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# 全局单例
# ─────────────────────────────────────────────────────────────────

_global_registry: Optional[OpRegistry] = None


def get_op_registry() -> OpRegistry:
    """
    获取全局 OpRegistry 单例。
    通常在 builtin_ops.register_builtin_ops() 调用后填充内置算子。
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = OpRegistry()
    return _global_registry


def reset_op_registry() -> None:
    """清空全局注册表（单元测试用）。"""
    global _global_registry
    _global_registry = None
