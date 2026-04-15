"""
通用 patch_model
================
遍历模型所有子模块，按各 OperatorDesc.inject_pattern 匹配并替换。

支持四种 inject_pattern 格式：
  1. ("attr",        attr_name,   type_substr)  → 替换属性（如 act_fn）
  2. ("module_type", type_substr)               → 替换整个 Module
  3. ("linear_name", name_substr)               → 替换特定名字的 nn.Linear
  4. callable(name: str, module) -> bool        → 自定义匹配

主要接口：
    patch_model(model, op_registry, fn_map) → dict[str, int]
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

import torch.nn as nn

if TYPE_CHECKING:
    from operators.op_desc import OperatorDesc
    from operators.op_registry import OpRegistry

logger = logging.getLogger(__name__)


def patch_model(
    model: nn.Module,
    op_registry: "OpRegistry",
    fn_map: dict[str, object],
) -> dict[str, int]:
    """
    通用模型算子注入：
      遍历 op_registry 中所有 forward OperatorDesc，
      按 inject_pattern 在模型中找到对应位置并替换。

    参数：
        model      — 目标 nn.Module
        op_registry — 已注册的算子描述
        fn_map     — build_custom_fn_map() 返回的注入函数/类字典

    返回：
        {op_name: replaced_count} — 每种算子的替换数量

    注意：
        - 保留对旧式 fn_map 键（"silu_fn" / "RMSNormModule"）的兼容
        - inject_pattern 为 None 的 desc 跳过（backward、仅生成不注入的算子）
    """
    replaced_counts: dict[str, int] = defaultdict(int)

    for desc in op_registry.get_forward_descs():
        if desc.inject_pattern is None:
            continue

        # 从 fn_map 里找到该算子对应的注入对象
        # 支持两种命名约定：
        #   新式：f"{desc.name}_forward" → fn_map 里存的是整个 "silu" 的前向函数
        #   旧式：f"{desc.name}_fn"  (silu) 或 "RMSNormModule" (rmsnorm)
        inject_obj = _find_inject_obj(desc, fn_map)
        if inject_obj is None:
            logger.debug(f"[patch_model] {desc.name}: fn_map 中无对应对象，跳过注入")
            continue

        pattern = desc.inject_pattern
        count = _apply_pattern(model, pattern, inject_obj, desc.name)
        replaced_counts[desc.name] += count

    result = dict(replaced_counts)
    summary = ", ".join(f"{k} × {v}" for k, v in result.items() if v > 0)
    if summary:
        logger.info(f"[patch_model] 注入完成: {summary}")
    else:
        logger.warning("[patch_model] 未替换任何算子，请检查 inject_pattern 配置")

    return result


def _find_inject_obj(
    desc: "OperatorDesc",
    fn_map: dict[str, object],
) -> Optional[object]:
    """
    从 fn_map 中找到 desc 对应的注入对象。

    查找优先级：
        1. fn_map[f"{desc.name}_fn"]          (新式，激活函数)
        2. fn_map[f"{desc.name.capitalize()}Module"]  (新式，Module 类)
        3. fn_map[f"{desc.name.upper()}Module"]
        4. fn_map["LinearModule"] 当 desc.name in ("matmul", "linear")  (GEMM 替换)
        5. fn_map[f"RMSNormModule"] 当 desc.name == "rmsnorm"  (旧式兼容)
        6. fn_map[f"{desc.name}"]
    """
    candidates = [
        f"{desc.name}_fn",
        f"{desc.name.capitalize()}Module",
        f"{desc.name.upper()}Module",
        desc.name,
    ]
    # 特殊兼容
    if desc.name == "rmsnorm":
        candidates.insert(0, "RMSNormModule")
    if desc.name == "silu":
        candidates.insert(0, "silu_fn")
    if desc.name in ("matmul", "linear", "gemm"):
        candidates.insert(0, "LinearModule")

    for key in candidates:
        if key in fn_map:
            return fn_map[key]
    return None


def _apply_pattern(
    model: nn.Module,
    pattern,
    inject_obj: object,
    op_name: str,
) -> int:
    """
    按 inject_pattern 格式在 model 中递归匹配并替换。
    返回替换数量。
    """
    count = 0

    if callable(pattern) and not isinstance(pattern, tuple):
        # 格式 4：自定义 callable
        count += _replace_by_callable(model, pattern, inject_obj)

    elif isinstance(pattern, tuple):
        ptype = pattern[0]

        if ptype == "attr":
            # 格式 1：("attr", attr_name, type_substr)
            _, attr_name, type_substr = pattern
            count += _replace_attr(model, attr_name, type_substr, inject_obj)

        elif ptype == "module_type":
            # 格式 2：("module_type", type_substr)
            _, type_substr = pattern
            count += _replace_module_type(model, type_substr, inject_obj)

        elif ptype == "linear_name":
            # 格式 3：("linear_name", name_substr)
            _, name_substr = pattern
            count += _replace_linear_by_name(model, name_substr, inject_obj)

        else:
            logger.warning(f"[patch_model] {op_name}: 未知 inject_pattern 类型 {ptype!r}")

    else:
        logger.warning(f"[patch_model] {op_name}: inject_pattern 格式无效: {pattern!r}")

    return count


# ─────────────────────────────────────────────────────────────────
# 四种替换策略
# ─────────────────────────────────────────────────────────────────

def _replace_attr(
    model: nn.Module,
    attr_name: str,
    type_substr: str,
    inject_obj: object,
) -> int:
    """格式1：替换 module.<attr_name>，类型名含 type_substr（大小写无关）。

    使用 object.__setattr__ 绕过 nn.Module 对子模块类型的限制，
    允许将属性替换为普通 callable（激活函数）。
    """
    count = 0
    for module in model.modules():
        if hasattr(module, attr_name):
            attr = getattr(module, attr_name)
            if type_substr.lower() in type(attr).__name__.lower():
                # 使用 object.__setattr__ 绕过 nn.Module 的类型检查
                # （act_fn 可以是 callable，不一定是 nn.Module）
                try:
                    object.__setattr__(module, attr_name, inject_obj)
                except Exception:
                    # 对于某些 nn.Module 子类，直接用 __dict__ 赋值
                    module.__dict__[attr_name] = inject_obj
                count += 1
    return count


def _replace_module_type(
    model: nn.Module,
    type_substr: str,
    inject_obj: object,
) -> int:
    """格式2：递归查找类型名含 type_substr 的子模块并替换。"""
    count = 0

    def _recurse(parent: nn.Module):
        nonlocal count
        for child_name, child in list(parent.named_children()):
            if type_substr.lower() in type(child).__name__.lower():
                # inject_obj 可能是类（需要实例化）或实例
                if isinstance(inject_obj, type):
                    # 尝试传递 weight 和 variance_epsilon（针对 RMSNorm）
                    try:
                        new_mod = inject_obj(child.weight, child.variance_epsilon)
                    except (AttributeError, TypeError):
                        try:
                            new_mod = inject_obj()
                        except Exception as e:
                            logger.warning(
                                f"[patch_model] 实例化 {type(inject_obj).__name__} 失败: {e}"
                            )
                            continue
                else:
                    new_mod = inject_obj
                setattr(parent, child_name, new_mod)
                count += 1
            else:
                _recurse(child)

    _recurse(model)
    return count


def _replace_linear_by_name(
    model: nn.Module,
    name_substr: str,
    inject_obj: object,
) -> int:
    """格式3：替换名字含 name_substr 的 nn.Linear（name_substr='' 匹配所有）。"""
    count = 0

    def _recurse(parent: nn.Module, prefix: str = ""):
        nonlocal count
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            is_linear = isinstance(child, nn.Linear)
            name_match = name_substr == "" or name_substr.lower() in child_name.lower()
            if is_linear and name_match:
                if isinstance(inject_obj, type):
                    try:
                        new_mod = inject_obj(child)
                    except TypeError:
                        new_mod = inject_obj(child.in_features, child.out_features,
                                             child.bias is not None)
                else:
                    new_mod = inject_obj
                setattr(parent, child_name, new_mod)
                count += 1
            else:
                _recurse(child, full_name)

    _recurse(model)
    return count


def _replace_by_callable(
    model: nn.Module,
    match_fn,
    inject_obj: object,
) -> int:
    """格式4：自定义匹配函数 match_fn(name, module) -> bool。"""
    count = 0

    def _recurse(parent: nn.Module, prefix: str = ""):
        nonlocal count
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if match_fn(full_name, child):
                if isinstance(inject_obj, type):
                    try:
                        new_mod = inject_obj(child)
                    except Exception:
                        new_mod = inject_obj()
                else:
                    new_mod = inject_obj
                setattr(parent, child_name, new_mod)
                count += 1
            else:
                _recurse(child, full_name)

    _recurse(model)
    return count
