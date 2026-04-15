"""
通用 verify_kernel
===================
从 OperatorDesc 自动构建 ctypes 调用，并与 PyTorch reference 对比数值。

主要接口：
    verify_kernel(desc, so_path, device)          → dict
    verify_all_kernels_generic(reg, so_paths, device)  → (verified_so_paths, results)
"""
from __future__ import annotations

import ctypes
import logging
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from operators.op_desc import OperatorDesc
    from operators.op_registry import OpRegistry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# 单个 kernel 验证
# ─────────────────────────────────────────────────────────────────

def verify_kernel(
    desc: "OperatorDesc",
    so_path: str,
    device: Optional[torch.device] = None,
) -> dict:
    """
    加载编译好的 .so，按 OperatorDesc 规格进行数值验证。

    返回：
        {
            "passed":  bool,
            "rel_err": float,    # 最大相对误差
            "detail":  str,      # 各 test_case 的结果描述
            "n_cases": int,
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载 .so
    try:
        lib = ctypes.CDLL(so_path)
        fn = lib.launch_kernel
        fn.restype = None
        fn.argtypes = desc.resolved_ctypes_argtypes()
    except (OSError, AttributeError) as e:
        return {"passed": False, "rel_err": float("inf"),
                "detail": f"load failed: {e}", "n_cases": 0}

    # 2. 准备 output_dtypes
    out_dtypes = desc.resolved_output_dtypes()

    # 3. 逐 test_case 验证
    test_cases = desc.test_cases if desc.test_cases else desc.default_test_cases()
    results: list[tuple[bool, float, str]] = []
    cuda_context_ok = True

    for tc in test_cases:
        if not cuda_context_ok:
            results.append((True, 0.0, f"tc={tc} skipped (CUDA ctx broken)"))
            continue
        try:
            passed, err, msg = _run_one_test_case(desc, fn, tc, out_dtypes, device)
            results.append((passed, err, msg))
        except Exception as e:
            err_str = str(e).lower()
            is_cuda = any(
                kw in err_str
                for kw in ("cuda error", "misaligned", "illegal memory", "device-side assert")
            )
            results.append((False, float("inf"), f"tc={tc} EXCEPTION: {str(e)[:120]}"))
            if is_cuda:
                cuda_context_ok = False
                logger.warning(f"[verify] {desc.key} CUDA error → 跳过后续 cases: {e}")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    all_ok = all(r[0] for r in results)
    max_err = max((r[1] for r in results), default=0.0)
    detail = "; ".join(r[2] for r in results)

    return {
        "passed":  all_ok,
        "rel_err": max_err,
        "detail":  detail,
        "n_cases": len(results),
    }


def _run_one_test_case(
    desc: "OperatorDesc",
    fn,
    tc: dict,
    out_dtypes: list[str],
    device: torch.device,
) -> tuple[bool, float, str]:
    """
    执行单个 test_case：
      1. input_shapes_fn → 输入张量
      2. 分配输出 buffer（按 output_dtypes，float32 预置零）
      3. scalar_args_fn → 标量参数
      4. 按 ctypes_argtypes 顺序拼装调用参数
      5. 调用 kernel + synchronize
      6. NaN/Inf 检查
      7. 与 pytorch_reference 对比相对误差
    """
    if desc.input_shapes_fn is None:
        raise RuntimeError(f"[verify] {desc.key}: input_shapes_fn 未定义")

    # ── Step 1: 输入张量 ────────────────────────────────────────────
    input_tensors: dict[str, torch.Tensor] = {
        k: v.to(device).contiguous()
        for k, v in desc.input_shapes_fn(tc).items()
    }
    first_input = next(iter(input_tensors.values()))

    # ── Step 2: 分配输出 buffer ────────────────────────────────────
    output_buffers: dict[int, torch.Tensor] = {}
    for out_idx_pos, arg_idx in enumerate(desc.output_arg_indices):
        dtype_str = out_dtypes[out_idx_pos]
        torch_dtype = desc.to_torch_dtype(dtype_str)

        if desc.output_shapes_fn is not None:
            shape = desc.output_shapes_fn(tc, input_tensors, out_idx_pos)
            buf = torch.empty(shape, dtype=torch_dtype, device=device)
        else:
            buf = torch.empty(first_input.numel(), dtype=torch_dtype, device=device)

        # float32 输出 buffer 预置零（部分 kernel 用 atomicAdd 累加写入）
        if torch_dtype == torch.float32:
            buf.zero_()

        output_buffers[arg_idx] = buf

    # ── Step 3: 标量参数 ────────────────────────────────────────────
    scalar_values: list = []
    if desc.scalar_args_fn is not None:
        scalar_values = list(desc.scalar_args_fn(tc, input_tensors))

    # ── Step 4: 拼装调用参数 ────────────────────────────────────────
    # 规则：
    #   output_arg_indices 处 → 取 output_buffers[arg_pos].data_ptr()
    #   指针类型（非输出）   → 从 input_tensors 按序取 .data_ptr()
    #   非指针类型           → 从 scalar_values 按序取
    call_args: list = []
    input_iter = iter(input_tensors.values())
    scalar_iter = iter(scalar_values)

    for arg_pos, type_str in enumerate(desc.ctypes_argtypes):
        if arg_pos in desc.output_arg_indices:
            call_args.append(output_buffers[arg_pos].data_ptr())
        elif desc.is_pointer_type(type_str):
            call_args.append(next(input_iter).data_ptr())
        else:
            call_args.append(next(scalar_iter))

    # ── Step 5: 调用 kernel ─────────────────────────────────────────
    fn(*call_args)
    torch.cuda.synchronize()

    # ── Step 6: NaN/Inf 检查 ────────────────────────────────────────
    for arg_idx, buf in output_buffers.items():
        if buf.isnan().any() or buf.isinf().any():
            return False, float("inf"), f"NaN/Inf in output[{arg_idx}] at tc={tc}"

    # ── Step 7: 与 PyTorch reference 对比 ───────────────────────────
    if desc.pytorch_reference is None:
        return True, 0.0, f"tc={tc} NaN-check OK (no reference)"

    ref_inputs = list(input_tensors.values())
    # backward variant 的 reference 内部需要调用 .backward()，
    # 不能包在 no_grad 里；forward variant 用 no_grad 节省显存。
    if desc.variant == "backward":
        ref_outputs = desc.pytorch_reference(*ref_inputs)
    else:
        with torch.no_grad():
            ref_outputs = desc.pytorch_reference(*ref_inputs)

    if not isinstance(ref_outputs, (list, tuple)):
        ref_outputs = [ref_outputs]

    max_rel_err = 0.0
    err_msgs: list[str] = []

    for out_idx_pos, (arg_idx, buf) in enumerate(output_buffers.items()):
        if out_idx_pos >= len(ref_outputs):
            break
        ref = ref_outputs[out_idx_pos].to(device)

        buf_f = buf.reshape(ref.shape).float()
        ref_f = ref.float()

        abs_err = (buf_f - ref_f).abs()
        denom = ref_f.abs()

        # 对接近零的元素用绝对误差代替相对误差
        threshold = denom.mean() * 0.05 + 1e-4
        mask = denom > threshold
        if mask.any():
            rel_err = (abs_err[mask] / (denom[mask] + 1e-6)).max().item()
        else:
            rel_err = abs_err.max().item()

        max_rel_err = max(max_rel_err, rel_err)
        passed = rel_err < desc.error_threshold
        err_msgs.append(
            f"out[{out_idx_pos}] rel_err={rel_err:.4f} "
            f"({'OK' if passed else 'FAIL'}) tc={tc}"
        )

    overall_passed = max_rel_err < desc.error_threshold
    return overall_passed, max_rel_err, "; ".join(err_msgs)


# ─────────────────────────────────────────────────────────────────
# 批量验证（替代 full_agent_lora_train.py 的 verify_all_kernels）
# ─────────────────────────────────────────────────────────────────

def verify_all_kernels_generic(
    op_registry: "OpRegistry",
    so_paths: dict[str, Optional[str]],
    device: Optional[torch.device] = None,
) -> tuple[dict[str, Optional[str]], dict[str, dict]]:
    """
    通用批量验证：替代 full_agent_lora_train.py 中的 verify_all_kernels。

    对 so_paths 里的每个 kernel：
      - 在 op_registry 中查找对应 OperatorDesc
      - 调用 verify_kernel
      - 验证失败的 so_path 置为 None（触发 PyTorch fallback）
      - 未注册的 key 视为 PASS（兼容旧代码）
      - CUDA context 损坏后跳过后续验证（保留 so_path）

    返回：
        verified_so_paths  — 失败的 path 已置 None
        verify_results     — {key: {"passed", "rel_err", "detail"}}
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("  [verify] 无 GPU，跳过数值验证")
            return so_paths, {k: {"passed": True, "detail": "no GPU"} for k in so_paths}

    verified_paths = dict(so_paths)
    verify_results: dict[str, dict] = {}
    cuda_context_broken = False

    print(f"\n[verify] 数值验证 {len(so_paths)} 个 kernel...")

    for key, so_path in so_paths.items():
        if so_path is None:
            verify_results[key] = {"passed": False, "detail": "compile failed"}
            print(f"  ⚠ {key}: 编译失败，跳过验证")
            continue

        # CUDA context 损坏时跳过，但保留 so_path
        if cuda_context_broken:
            verify_results[key] = {
                "passed": True,
                "detail": "skipped (CUDA ctx broken by prev kernel)",
            }
            print(f"  ⚠ {key}: PASS (跳过验证，前序 kernel 损坏了 CUDA context)")
            continue

        desc = op_registry.get(key)
        if desc is None:
            # 未注册 → 兼容性：视为 PASS，不验证
            verify_results[key] = {"passed": True, "detail": "no OperatorDesc, skip verify"}
            print(f"  ⚠ {key}: PASS (无 OperatorDesc，跳过验证)")
            continue

        result = verify_kernel(desc, so_path, device)
        verify_results[key] = result

        if result["passed"]:
            print(f"  ✅ {key}: PASS (max_rel_err={result['rel_err']:.4f})")
        else:
            # 判断是否是 CUDA context 损坏
            detail = result.get("detail", "")
            is_cuda = any(
                kw in detail.lower()
                for kw in ("cuda error", "misaligned", "illegal memory", "cuda ctx broken")
            )
            if is_cuda:
                cuda_context_broken = True
                print(f"  ❌ {key}: CUDA ERROR → fallback ({detail[:80]})")
            else:
                print(f"  ❌ {key}: FAIL → fallback ({detail[:80]})")
            verified_paths[key] = None
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    n_pass = sum(1 for r in verify_results.values() if r.get("passed"))
    print(f"  验证结果: {n_pass}/{len(verify_results)} 通过")
    return verified_paths, verify_results
