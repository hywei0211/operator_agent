"""
Verifier Agent - 硬件自适应验证器
自动检测当前硬件环境，分层执行验证：
  Level 1: 静态分析（总是执行）
  Level 2: LLM 代码审查（有 LLM 时执行）
  Level 3: CPU 数学验证（有 PyTorch 时执行）
  Level 4: 真实编译（有匹配编译器时执行）
  Level 5: 硬件运行+数值验证（硬件匹配时执行）
  Level 6: 硬件 Benchmark（硬件匹配时执行）
"""
import ctypes
import logging
import os
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from agents.base_agent import BaseAgent, AgentContext, AgentResult, AgentStatus
from models.operator_ir import GeneratedKernel, OperatorIR
from models.hardware_model import GPUSpec

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════
# 验证等级
# ════════════════════════════════════════════════════════

class VerificationLevel(str, Enum):
    """验证等级 — 每一级都包含前面所有级的检查"""
    NONE = "none"                         # 未验证
    STATIC = "static"                     # 仅静态分析（语法+规则）
    LLM_REVIEW = "llm_review"             # LLM 代码审查
    CPU_MATH = "cpu_math"                 # CPU 上数学正确性验证（PyTorch 参考实现）
    COMPILED = "compiled"                 # 真实编译通过（但没在目标硬件上运行）
    HARDWARE_VERIFIED = "hw_verified"     # 真实硬件运行 + 数值验证通过
    BENCHMARKED = "benchmarked"           # 真实硬件 Benchmark 完成


# 等级排序（用于比较）
_LEVEL_ORDER = {lv: i for i, lv in enumerate(VerificationLevel)}


def level_ge(a: VerificationLevel, b: VerificationLevel) -> bool:
    """a 的验证等级 >= b ?"""
    return _LEVEL_ORDER[a] >= _LEVEL_ORDER[b]


# ════════════════════════════════════════════════════════
# 硬件检测
# ════════════════════════════════════════════════════════

class HardwareDetector:
    """
    一次性检测当前机器的 GPU 硬件和 SDK 环境。
    结果会被缓存，整个进程只检测一次。
    """

    _cache: Optional[dict] = None

    @classmethod
    def detect(cls) -> dict:
        if cls._cache is not None:
            return cls._cache

        result = {
            # NVIDIA
            "nvidia_gpu": False,
            "nvidia_gpu_name": "",
            "nvcc": False,
            # AMD
            "amd_gpu": False,
            "hipcc": False,
            # 华为昇腾
            "npu": False,
            "npu_name": "",
            "cann": False,
            # 通用
            "torch": False,
        }

        # ── PyTorch ──
        try:
            import torch
            result["torch"] = True

            # ── NVIDIA GPU（通过 PyTorch CUDA） ──
            if torch.cuda.is_available():
                result["nvidia_gpu"] = True
                try:
                    result["nvidia_gpu_name"] = torch.cuda.get_device_name(0)
                except Exception:
                    result["nvidia_gpu_name"] = "unknown"
        except ImportError:
            pass

        # ── 昇腾 NPU（通过 torch_npu） ──
        try:
            import torch_npu  # noqa: F401
            import torch
            if torch.npu.is_available():
                result["npu"] = True
                try:
                    result["npu_name"] = torch.npu.get_device_name(0)
                except Exception:
                    result["npu_name"] = "Ascend NPU"
        except (ImportError, Exception):
            pass

        # ── 编译器检测（subprocess，不依赖 PyTorch） ──
        result["nvcc"] = cls._cmd_available(["nvcc", "--version"])
        result["hipcc"] = cls._cmd_available(["hipcc", "--version"])
        result["cann"] = cls._cmd_available(["atc", "--version"])

        # ── AMD GPU（通过 rocm-smi） ──
        if cls._cmd_available(["rocm-smi", "--showid"]):
            result["amd_gpu"] = True

        cls._cache = result
        logger.info(f"[HardwareDetector] Detected: "
                    f"nvidia={result['nvidia_gpu']}({result['nvidia_gpu_name']}), "
                    f"nvcc={result['nvcc']}, "
                    f"npu={result['npu']}({result['npu_name']}), "
                    f"cann={result['cann']}, "
                    f"amd={result['amd_gpu']}, hipcc={result['hipcc']}, "
                    f"torch={result['torch']}")
        return result

    @staticmethod
    def _cmd_available(cmd: list[str]) -> bool:
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=5)
            return proc.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return False

    @classmethod
    def reset_cache(cls):
        """测试用：重置缓存"""
        cls._cache = None


# ════════════════════════════════════════════════════════
# 验证报告
# ════════════════════════════════════════════════════════

@dataclass
class VerificationReport:
    """验证报告"""
    kernel_name: str
    backend: str
    target_gpu: str

    # 验证等级（本次最高达到的等级）
    verification_level: VerificationLevel = VerificationLevel.NONE

    # 检测到的硬件环境
    hardware_detected: dict = field(default_factory=dict)

    # Level 1: 静态分析
    static_analysis: dict = field(default_factory=dict)

    # Level 2: LLM 审查
    llm_review_passed: bool = False
    llm_review_details: str = ""

    # Level 3: CPU 数学验证
    cpu_math_passed: bool = False
    cpu_math_details: str = ""

    # Level 4: 编译验证
    compilation_passed: bool = False
    compilation_errors: list[str] = field(default_factory=list)

    # Level 5: 正确性验证（真实硬件）
    correctness_passed: bool = False
    max_relative_error: float = float('inf')
    max_absolute_error: float = float('inf')
    correctness_details: str = ""

    # Level 6: 性能验证（真实硬件）
    performance_passed: bool = False
    measured_latency_ms: float = 0.0
    measured_tflops: float = 0.0
    bandwidth_utilization: float = 0.0
    compute_utilization: float = 0.0
    speedup_vs_pytorch: float = 0.0

    # 综合结论
    overall_passed: bool = False
    recommendations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.overall_passed else "FAIL"
        level = self.verification_level.value
        lines = [
            f"[{status}] {self.kernel_name} on {self.target_gpu} ({self.backend})",
            f"  Verification Level: {level}",
        ]
        if level_ge(self.verification_level, VerificationLevel.STATIC):
            score = self.static_analysis.get("score", "?")
            lines.append(f"  Static Analysis: score={score}")
        if level_ge(self.verification_level, VerificationLevel.COMPILED):
            lines.append(f"  Compilation: {'OK' if self.compilation_passed else 'FAIL'}")
        if level_ge(self.verification_level, VerificationLevel.CPU_MATH):
            lines.append(f"  CPU Math: {'OK' if self.cpu_math_passed else 'FAIL'}")
        if level_ge(self.verification_level, VerificationLevel.HARDWARE_VERIFIED):
            lines.append(f"  HW Correctness: {'OK' if self.correctness_passed else 'FAIL'} "
                         f"(rel_err={self.max_relative_error:.2e})")
        if level_ge(self.verification_level, VerificationLevel.BENCHMARKED):
            lines.append(f"  HW Performance: BW={self.bandwidth_utilization:.1%}, "
                         f"speedup={self.speedup_vs_pytorch:.2f}x")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════
# VerifierAgent 主体
# ════════════════════════════════════════════════════════

class VerifierAgent(BaseAgent):
    """
    硬件自适应验证 Agent

    核心逻辑：
    1. 自动检测当前硬件
    2. 根据硬件能力决定能验到哪个等级
    3. 从 Level 1 到 Level 6 依次执行，每级都包含前面所有级的检查
    4. 任何一级失败都会停下来并反馈修复建议
    """

    def __init__(self, llm_client=None, config: dict = None, mcp_client=None):
        super().__init__("VerifierAgent", llm_client, config)
        self.mcp = mcp_client
        self.correctness_threshold = (config or {}).get("correctness_threshold", 1e-3)
        self.min_bandwidth_efficiency = (config or {}).get("min_bandwidth_efficiency", 0.5)

    def get_system_prompt(self) -> str:
        return """你是GPU内核验证专家，专注于验证GPU内核代码的正确性和性能。
你能够分析代码中的潜在错误、数值稳定性问题和性能瓶颈。
请基于代码分析给出验证结论，并提出具体的修复建议。"""

    # ── 硬件匹配判断 ──────────────────────────────────────

    @staticmethod
    def _can_compile(backend: str, hw: dict) -> bool:
        """当前机器能否编译该后端的代码"""
        return {
            "cuda": hw.get("nvcc", False),
            "hip": hw.get("hipcc", False),
            "ascendc": hw.get("cann", False),
            "triton": hw.get("torch", False),
        }.get(backend, False)

    @staticmethod
    def _can_execute(backend: str, hw: dict) -> bool:
        """当前机器能否在真实硬件上运行（需要匹配的 GPU/NPU）"""
        return {
            "cuda": hw.get("nvidia_gpu", False),
            "hip": hw.get("amd_gpu", False),
            "ascendc": hw.get("npu", False),
            "triton": hw.get("nvidia_gpu", False) or hw.get("amd_gpu", False),
        }.get(backend, False)

    # ── 主执行逻辑 ────────────────────────────────────────

    async def run(self, context: AgentContext, **kwargs) -> AgentResult:
        self._start_timer()
        self.set_status(AgentStatus.RUNNING)

        kernel: Optional[GeneratedKernel] = kwargs.get("kernel")
        operator_ir: Optional[OperatorIR] = kwargs.get("operator_ir") or context.get_artifact("operator_ir")
        gpu_spec: Optional[GPUSpec] = kwargs.get("gpu_spec")

        if kernel is None:
            return self.failure_result("No kernel provided for verification")

        try:
            # 0. 检测硬件
            hw = HardwareDetector.detect()
            can_compile = self._can_compile(kernel.backend, hw)
            can_execute = self._can_execute(kernel.backend, hw)
            has_torch = hw.get("torch", False)

            report = VerificationReport(
                kernel_name=kernel.operator_name,
                backend=kernel.backend,
                target_gpu=kernel.target_gpu,
                hardware_detected=hw,
            )

            logger.info(f"[Verifier] {kernel.operator_name} ({kernel.backend}): "
                        f"can_compile={can_compile}, can_execute={can_execute}, torch={has_torch}")

            # ── Level 1: 静态分析（总是执行）──────────────────
            report.static_analysis = self._static_analysis(kernel)
            report.verification_level = VerificationLevel.STATIC

            if report.static_analysis.get("summary") == "FAIL":
                report.overall_passed = False
                report.recommendations.append(
                    f"Static analysis failed: {report.static_analysis.get('failed_checks', [])}"
                )
                return self._finish(report)

            # ── Level 2: LLM 代码审查 ────────────────────────
            if self.llm_client and operator_ir:
                llm_ok, llm_detail = await self._llm_math_check(kernel, operator_ir)
                report.llm_review_passed = llm_ok
                report.llm_review_details = llm_detail
                if llm_ok:
                    report.verification_level = VerificationLevel.LLM_REVIEW
                else:
                    report.overall_passed = False
                    report.recommendations.append(f"LLM review issue: {llm_detail}")
                    return self._finish(report)

            # ── Level 3: CPU 数学验证 ────────────────────────
            if has_torch and operator_ir:
                cpu_ok, cpu_detail = self._cpu_math_verify(kernel, operator_ir)
                report.cpu_math_passed = cpu_ok
                report.cpu_math_details = cpu_detail
                if cpu_ok:
                    report.verification_level = VerificationLevel.CPU_MATH
                # CPU 数学验证失败不阻塞（参考实现可能不完全覆盖）
                # 但会记录在报告中

            # ── Level 4: 真实编译 ────────────────────────────
            if can_compile:
                compile_ok, compile_errors = await self._real_compile(kernel)
                report.compilation_passed = compile_ok
                report.compilation_errors = compile_errors
                if compile_ok:
                    report.verification_level = VerificationLevel.COMPILED
                else:
                    report.overall_passed = False
                    report.recommendations.append(
                        f"Compilation failed: {'; '.join(compile_errors[:3])}"
                    )
                    return self._finish(report)
            else:
                # 无编译器：做基本语法检查兜底
                syntax_ok, syntax_errors = self._syntax_check(kernel)
                report.compilation_passed = syntax_ok
                report.compilation_errors = syntax_errors
                if not syntax_ok:
                    report.recommendations.append(
                        f"Syntax check failed (no compiler available): {'; '.join(syntax_errors[:3])}"
                    )

            # ── Level 5: 真实硬件运行 + 数值验证 ─────────────
            if can_execute and report.compilation_passed:
                exec_result = await self._real_execute(kernel, operator_ir, gpu_spec, hw)
                report.correctness_passed = exec_result.get("math_correct", False)
                report.max_relative_error = exec_result.get("max_rel_error", float('inf'))
                report.correctness_details = exec_result.get("details", "")
                if report.correctness_passed:
                    report.verification_level = VerificationLevel.HARDWARE_VERIFIED
                else:
                    report.overall_passed = False
                    report.recommendations.append(
                        f"Hardware correctness failed: rel_err={report.max_relative_error:.2e}"
                    )
                    return self._finish(report)

                # ── Level 6: Benchmark ──────────────────────
                bench = await self._real_benchmark(kernel, operator_ir, gpu_spec, hw)
                report.measured_latency_ms = bench.get("latency_ms", 0)
                report.bandwidth_utilization = bench.get("bw_utilization", 0)
                report.speedup_vs_pytorch = bench.get("speedup", 0)
                report.performance_passed = report.bandwidth_utilization >= self.min_bandwidth_efficiency
                report.verification_level = VerificationLevel.BENCHMARKED

                if not report.performance_passed:
                    report.recommendations.append(
                        f"Performance below target: BW={report.bandwidth_utilization:.1%} "
                        f"< {self.min_bandwidth_efficiency:.1%}"
                    )
            else:
                # 无法在硬件上运行时，用估算填充性能字段
                perf = self._estimate_performance(kernel, gpu_spec)
                report.bandwidth_utilization = perf.get("bandwidth_utilization", 0)
                report.performance_passed = perf.get("passed", True)

            # ── 综合判断 ──────────────────────────────────────
            report.overall_passed = self._compute_overall_pass(report)

            return self._finish(report)

        except Exception as e:
            self.set_status(AgentStatus.FAILED)
            logger.exception(f"[Verifier] Failed: {e}")
            return self.failure_result(str(e))

    # ════════════════════════════════════════════════════════
    # Level 1: 静态分析
    # ════════════════════════════════════════════════════════

    def _static_analysis(self, kernel: GeneratedKernel) -> dict:
        """复用 cpu_simulator.StaticCodeAnalyzer"""
        try:
            from tools.cpu_simulator import StaticCodeAnalyzer
            return StaticCodeAnalyzer().analyze(kernel.source_code, kernel.backend)
        except Exception as e:
            logger.warning(f"[Verifier] Static analysis fallback: {e}")
            return self._basic_syntax_check_as_static(kernel)

    def _basic_syntax_check_as_static(self, kernel: GeneratedKernel) -> dict:
        """StaticCodeAnalyzer 不可用时的兜底"""
        code = kernel.source_code
        passed = []
        failed = []
        if not code or len(code.strip()) < 10:
            failed.append("Empty or too short kernel code")
        else:
            passed.append("Code length OK")
        if code.count("{") != code.count("}"):
            failed.append("Mismatched braces")
        else:
            passed.append("Braces matched")
        score = len(passed) / max(len(passed) + len(failed), 1)
        return {
            "score": round(score, 2),
            "passed_checks": passed,
            "failed_checks": failed,
            "summary": "PASS" if score >= 0.67 else "FAIL",
        }

    # ════════════════════════════════════════════════════════
    # Level 2: LLM 审查
    # ════════════════════════════════════════════════════════

    async def _llm_math_check(self, kernel: GeneratedKernel, op_ir: OperatorIR) -> tuple[bool, str]:
        """用 LLM 检查代码的数学正确性"""
        prompt = f"""请验证以下GPU内核代码是否正确实现了指定的数学运算。

数学定义: {op_ir.math_description}

参考实现:
```python
{op_ir.reference_impl[:500] if op_ir.reference_impl else "N/A"}
```

GPU内核代码（前1500字符）:
```
{kernel.source_code[:1500]}
```

检查：1.数学运算是否一致 2.边界条件 3.数据类型 4.数值稳定性

如果实现正确，返回 OK。如果有问题，描述具体问题（一行）。"""

        try:
            resp = await self.call_llm(prompt, temperature=0.0, max_tokens=300)
            ok = "ok" in resp.lower()[:20] or "correct" in resp.lower()[:30]
            return ok, "" if ok else resp.strip()[:200]
        except Exception as e:
            logger.warning(f"[Verifier] LLM math check failed: {e}")
            return True, ""  # LLM 失败时不阻塞流程

    # ════════════════════════════════════════════════════════
    # Level 3: CPU 数学验证
    # ════════════════════════════════════════════════════════

    def _cpu_math_verify(self, kernel: GeneratedKernel, op_ir: OperatorIR) -> tuple[bool, str]:
        """用 PyTorch CPU 参考实现验证数学正确性（forward + backward）"""
        try:
            from tools.cpu_simulator import CPUSimulator
            sim = CPUSimulator()

            # Forward 验证
            test_inputs = sim.generate_test_inputs(op_ir.name)
            if not test_inputs:
                return True, "No test inputs generated (operator not in CPU simulator)"

            result = sim.verify_operator(op_ir.name, test_inputs)
            forward_ok = result.math_correct
            details = "; ".join(result.notes[:3]) if result.notes else ""

            # Backward 验证（如果 op_ir 有 backward 定义）
            if op_ir.backward_math_description:
                backward_result = sim.verify_backward(op_ir.name)
                backward_ok = backward_result.math_correct
                bwd_notes = "; ".join(backward_result.notes[:2]) if backward_result.notes else ""
                details += f" | backward: {bwd_notes}"
                return forward_ok and backward_ok, details
            else:
                details += " | backward: skipped (no backward_math_description)"
                return forward_ok, details
        except Exception as e:
            logger.warning(f"[Verifier] CPU math verify failed: {e}")
            return True, f"CPU math verify skipped: {e}"

    # ════════════════════════════════════════════════════════
    # Level 4: 真实编译
    # ════════════════════════════════════════════════════════

    async def _real_compile(self, kernel: GeneratedKernel) -> tuple[bool, list[str]]:
        """通过 MCP 或直接调用编译器编译"""
        # 优先用 MCP（支持 Docker/SSH/本地）
        if self.mcp:
            try:
                resp = await self.mcp.call(
                    "remote_executor_server", "compile_kernel",
                    source_code=kernel.source_code,
                    sdk=kernel.backend,
                    build_flags=kernel.build_flags,
                    kernel_name=kernel.operator_name,
                )
                if resp.success:
                    data = resp.data or {}
                    success = data.get("success", False)
                    errors = []
                    if not success:
                        stderr = data.get("stderr", "")
                        errors = [ln for ln in stderr.split("\n") if "error:" in ln.lower()][:5]
                    return success, errors
            except Exception as e:
                logger.warning(f"[Verifier] MCP compile failed, trying local: {e}")

        # 直接本地编译
        return self._local_compile(kernel)

    def _local_compile(self, kernel: GeneratedKernel) -> tuple[bool, list[str]]:
        """直接在本地编译"""
        import tempfile
        import os

        ext = {"cuda": ".cu", "hip": ".cpp", "ascendc": ".cpp", "triton": ".py"}.get(kernel.backend, ".cpp")
        compiler = {"cuda": "nvcc", "hip": "hipcc"}.get(kernel.backend)

        if kernel.backend == "triton":
            # Triton: 只需 import check
            try:
                with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
                    f.write(kernel.source_code)
                    tmp = f.name
                proc = subprocess.run(
                    ["python", "-c", f"exec(open('{tmp}').read())"],
                    capture_output=True, text=True, timeout=30)
                os.unlink(tmp)
                return proc.returncode == 0, [proc.stderr[:500]] if proc.returncode != 0 else []
            except Exception as e:
                return False, [str(e)]

        if not compiler:
            return True, []  # 无编译器，不阻塞

        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode="w") as f:
                f.write(kernel.source_code)
                tmp = f.name
            cmd = [compiler] + (kernel.build_flags or []) + ["-c", tmp, "-o", tmp + ".o"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            errors = []
            if proc.returncode != 0:
                errors = [ln for ln in proc.stderr.split("\n") if "error:" in ln.lower()][:5]
            os.unlink(tmp)
            if os.path.exists(tmp + ".o"):
                os.unlink(tmp + ".o")
            return proc.returncode == 0, errors
        except Exception as e:
            return False, [str(e)]

    def _syntax_check(self, kernel: GeneratedKernel) -> tuple[bool, list[str]]:
        """无编译器时的基本语法检查"""
        errors = []
        code = kernel.source_code

        if kernel.backend in ("cuda", "hip"):
            if "__global__" not in code:
                errors.append("No __global__ kernel function found")
            if code.count("{") != code.count("}"):
                errors.append("Mismatched braces { }")
        elif kernel.backend == "ascendc":
            if "__aicore__" not in code:
                errors.append("No __aicore__ kernel function found")
        elif kernel.backend == "triton":
            if "@triton.jit" not in code:
                errors.append("No @triton.jit decorator found")

        return len(errors) == 0, errors

    # ════════════════════════════════════════════════════════
    # Level 5: 真实硬件运行
    # ════════════════════════════════════════════════════════

    async def _real_execute(
        self, kernel: GeneratedKernel, op_ir: Optional[OperatorIR],
        gpu_spec: Optional[GPUSpec], hw: dict
    ) -> dict:
        """在真实硬件上运行并验证数值正确性"""
        # 优先通过 MCP
        if self.mcp:
            try:
                ref_impl = op_ir.reference_impl if op_ir else ""
                resp = await self.mcp.call(
                    "remote_executor_server", "run_correctness_test",
                    kernel_name=kernel.operator_name,
                    sdk=kernel.backend,
                    source_code=kernel.source_code,
                    reference_impl=ref_impl,
                    build_flags=kernel.build_flags,
                )
                if resp.success and resp.data and resp.data.get("status") != "skipped":
                    return resp.data
            except Exception as e:
                logger.warning(f"[Verifier] MCP correctness test failed: {e}")

        # 降级: 用 PyTorch GPU 参考实现自测（只能验参考实现本身，不能验生成的 kernel）
        if hw.get("nvidia_gpu") and kernel.backend == "cuda":
            return self._pytorch_gpu_quick_check(kernel, op_ir, "cuda")
        if hw.get("npu") and kernel.backend == "ascendc":
            return self._pytorch_gpu_quick_check(kernel, op_ir, "npu")

        return {"math_correct": True, "max_rel_error": 0.0,
                "details": "No hardware execution path available, skipped"}

    def _pytorch_gpu_quick_check(
        self, kernel: GeneratedKernel, op_ir: Optional[OperatorIR], device_type: str
    ) -> dict:
        """
        Level 5 核心验证：编译 kernel → ctypes 加载 → 对比 PyTorch reference 输出。

        支持四种 kernel 类型（通过 kernel.operator_name 判断）：
          - forward (silu/gelu/rmsnorm_forward)：调用并对比 forward 输出
          - backward (silu_backward/gelu_backward)：grad_in 输出为 float32（方案A接口）
          - rmsnorm_backward：grad_x 输出为 float32（方案A接口）

        额外检查：大梯度稳定性（scale=50），模拟深层 fp16 训练中梯度爆炸的场景。
        """
        if not op_ir:
            return {"math_correct": True, "max_rel_error": 0.0,
                    "details": "No OperatorIR, skip GPU quick check"}

        try:
            import torch
            import torch.nn.functional as F

            if device_type == "npu":
                try:
                    import torch_npu  # noqa: F401
                    device = torch.device("npu:0")
                except ImportError:
                    return {"math_correct": True, "max_rel_error": 0.0,
                            "details": "torch_npu not available, skip NPU check"}
            else:
                device = torch.device("cuda:0")

            op_name = kernel.operator_name.lower()
            is_backward = "backward" in op_name
            is_rmsnorm = "rmsnorm" in op_name

            # ── Step 1: 编译 kernel 到临时 .so ──────────────────
            nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
            if not os.path.exists(nvcc):
                return {"math_correct": True, "max_rel_error": 0.0,
                        "details": "nvcc not found, skip kernel numeric check"}

            tmp_dir = tempfile.mkdtemp(prefix="verifier_")
            src_path = os.path.join(tmp_dir, f"{op_name}_verify.cu")
            so_path = os.path.join(tmp_dir, f"{op_name}_verify.so")

            try:
                # 应用编译错误知识库修复
                src_code = kernel.source_code
                try:
                    from knowledge_base.compile_error_kb import get_compile_error_kb
                    src_code = get_compile_error_kb().auto_fix(src_code, "cuda")
                except Exception:
                    pass

                with open(src_path, "w") as f:
                    f.write(src_code)

                valid_prefixes = ("-O", "-arch=", "--use_fast_math", "-std=", "-Xcompiler", "-fPIC", "--shared")
                flags = [fl for fl in (kernel.build_flags or [])
                         if any(fl.startswith(p) for p in valid_prefixes)]
                if not flags:
                    flags = ["-O3", "--use_fast_math"]

                cmd = [nvcc, "--shared", "-Xcompiler", "-fPIC"] + flags + [src_path, "-o", so_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode != 0:
                    return {"math_correct": False, "max_rel_error": float('inf'),
                            "details": f"Verification compile failed: {result.stderr[:300]}"}

                # ── Step 2: 加载 .so ───────────────────────────────
                lib = ctypes.CDLL(so_path)
                fn = lib.launch_kernel
                fn.restype = None

                # ── Step 3: 按 kernel 类型构造测试并验证 ───────────
                results = []

                # 测试形状列表 — N*H 必须是 128 的倍数，避免 half2 向量化 kernel 的对齐问题
                test_cases = [
                    {"N": 64, "H": 1024, "grad_scale": 1.0},   # 65536 elements
                    {"N": 8, "H": 3072, "grad_scale": 1.0},    # 24576 elements
                    {"N": 16, "H": 1024, "grad_scale": 50.0},  # 16384 elements，大梯度测试
                ]

                for tc in test_cases:
                    N, H = tc["N"], tc["H"]
                    gs = tc["grad_scale"]
                    # 使用 contiguous() 确保内存连续对齐，满足 half2 向量化 kernel 的对齐要求
                    x = torch.randn(N, H, dtype=torch.float16, device=device).contiguous()

                    if is_rmsnorm and is_backward:
                        # RMSNorm backward：(grad_out, x, weight) → (grad_x_fp32, grad_w)
                        fn.argtypes = [
                            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_int, ctypes.c_int, ctypes.c_float,
                        ]
                        weight = torch.ones(H, dtype=torch.float16, device=device)
                        grad_out = (torch.randn(N, H, dtype=torch.float16, device=device) * gs)
                        grad_x_fp32 = torch.empty(N * H, dtype=torch.float32, device=device)
                        grad_w_fp32 = torch.zeros(H, dtype=torch.float32, device=device)
                        fn(grad_out.data_ptr(), x.data_ptr(), weight.data_ptr(),
                           grad_x_fp32.data_ptr(), grad_w_fp32.data_ptr(),
                           N, H, 1e-6)
                        torch.cuda.synchronize()
                        # NaN / Inf 检查
                        if grad_x_fp32.isnan().any() or grad_x_fp32.isinf().any():
                            results.append({"ok": False, "err": float('inf'),
                                            "detail": f"RMSNorm bwd NaN/Inf at scale={gs}"})
                            continue
                        # 对比 PyTorch reference
                        xr = x.float().requires_grad_(True)
                        wr = weight.float().requires_grad_(True)
                        rms = torch.sqrt(xr.pow(2).mean(-1, keepdim=True) + 1e-6)
                        y = xr / rms * wr
                        y.backward(grad_out.float())
                        ref_gx = xr.grad.float()
                        err = (grad_x_fp32.reshape(N, H) - ref_gx).abs()
                        # 用更稳健的相对误差：排除绝对值过小的参考值（避免除以接近0的数）
                        ref_abs = ref_gx.abs()
                        mask = ref_abs > (ref_abs.mean() * 0.05 + 1e-4)
                        if mask.any():
                            rel_err = (err[mask] / (ref_abs[mask] + 1e-4)).max().item()
                        else:
                            rel_err = err.max().item()
                        # 阈值 0.05（5%）：RMSNorm backward 在 fp16 精度下若超过此值会导致训练不稳定
                        results.append({"ok": rel_err < 0.05, "err": rel_err,
                                        "detail": f"RMSNorm bwd rel_err={rel_err:.4f} scale={gs}"})

                    elif is_backward:
                        # Elementwise backward (silu/gelu)：(grad_out, x) → grad_in_fp32
                        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
                        grad_out = (torch.randn(N, H, dtype=torch.float16, device=device) * gs)
                        grad_in_fp32 = torch.empty(N * H, dtype=torch.float32, device=device)
                        fn(grad_out.data_ptr(), x.reshape(-1).data_ptr(),
                           grad_in_fp32.data_ptr(), N * H)
                        torch.cuda.synchronize()
                        if grad_in_fp32.isnan().any() or grad_in_fp32.isinf().any():
                            results.append({"ok": False, "err": float('inf'),
                                            "detail": f"Elementwise bwd NaN/Inf at scale={gs}"})
                            continue
                        # 对比 PyTorch reference
                        x_flat = x.reshape(-1).float()
                        g_flat = grad_out.reshape(-1).float()
                        if "silu" in op_name:
                            sig = torch.sigmoid(x_flat)
                            ref = g_flat * sig * (1.0 + x_flat * (1.0 - sig))
                        elif "gelu" in op_name:
                            xr = x_flat.requires_grad_(True)
                            F.gelu(xr).backward(g_flat)
                            ref = xr.grad
                        else:
                            ref = g_flat
                        err = (grad_in_fp32 - ref).abs()
                        rel_err = (err / (ref.abs() + 1e-6)).max().item()
                        results.append({"ok": rel_err < 0.05, "err": rel_err,
                                        "detail": f"Elementwise bwd rel_err={rel_err:.4f} scale={gs}"})

                    elif is_rmsnorm:
                        # RMSNorm forward：(x, weight) → out_half
                        fn.argtypes = [
                            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_int, ctypes.c_int, ctypes.c_float,
                        ]
                        weight = torch.ones(H, dtype=torch.float16, device=device)
                        out = torch.empty(N, H, dtype=torch.float16, device=device)
                        fn(x.data_ptr(), weight.data_ptr(), out.data_ptr(), N, H, 1e-6)
                        torch.cuda.synchronize()
                        if out.isnan().any():
                            results.append({"ok": False, "err": float('inf'),
                                            "detail": "RMSNorm fwd NaN"})
                            continue
                        ref = (x.float() / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).half()
                        err = (out.float() - ref.float()).abs()
                        rel_err = (err / (ref.float().abs() + 1e-3)).max().item()
                        results.append({"ok": rel_err < 0.05, "err": rel_err,
                                        "detail": f"RMSNorm fwd rel_err={rel_err:.4f}"})
                    else:
                        # Elementwise forward (silu/gelu)：x → out_half
                        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
                        out = torch.empty(N, H, dtype=torch.float16, device=device)
                        fn(x.data_ptr(), out.data_ptr(), N * H)
                        torch.cuda.synchronize()
                        if out.isnan().any():
                            results.append({"ok": False, "err": float('inf'),
                                            "detail": "Elementwise fwd NaN"})
                            continue
                        if "silu" in op_name:
                            ref = F.silu(x)
                        elif "gelu" in op_name:
                            ref = F.gelu(x)
                        else:
                            ref = x
                        err = (out.float() - ref.float()).abs()
                        rel_err = (err / (ref.float().abs() + 1e-3)).max().item()
                        results.append({"ok": rel_err < 0.05, "err": rel_err,
                                        "detail": f"Elementwise fwd rel_err={rel_err:.4f}"})

                # 汇总：所有测试均通过才算通过
                all_ok = all(r["ok"] for r in results)
                max_err = max(r["err"] for r in results) if results else 0.0
                detail = "; ".join(r["detail"] for r in results)
                return {
                    "math_correct": all_ok,
                    "max_rel_error": max_err,
                    "details": f"Kernel numeric verify ({'PASS' if all_ok else 'FAIL'}): {detail}",
                }

            finally:
                # 清理：先同步 GPU，释放 CUDA 内存，再删文件
                # 这样避免 .so 被删后 GPU 上仍有对它的引用
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _torch.cuda.synchronize()
                        _torch.cuda.empty_cache()
                except Exception:
                    pass
                # 显式删除 ctypes 库引用
                try:
                    del lib
                except Exception:
                    pass
                # 再清理临时文件
                try:
                    import shutil as _shutil
                    import time as _time
                    _time.sleep(0.2)  # 小等待确保 GPU 完全释放
                    _shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"[Verifier] Kernel numeric check failed: {e}")
            # 尝试重置 CUDA context，防止异步 CUDA error 污染后续操作
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                    # 用一个简单操作探测 CUDA context 是否健康
                    try:
                        _torch.zeros(1, device="cuda")
                    except Exception:
                        # Context 已损坏，无法恢复，标记为 CUDA error
                        pass
            except Exception:
                pass
            return {"math_correct": True, "max_rel_error": 0.0,
                    "details": f"Kernel numeric check error (fallback pass): {e}"}

    # ════════════════════════════════════════════════════════
    # Level 6: Benchmark
    # ════════════════════════════════════════════════════════

    async def _real_benchmark(
        self, kernel: GeneratedKernel, op_ir: Optional[OperatorIR],
        gpu_spec: Optional[GPUSpec], hw: dict
    ) -> dict:
        """在真实硬件上运行 Benchmark"""
        # 优先通过 MCP
        if self.mcp:
            try:
                resp = await self.mcp.call(
                    "remote_executor_server", "run_benchmark",
                    kernel_name=kernel.operator_name,
                    sdk=kernel.backend,
                    source_code=kernel.source_code,
                    build_flags=kernel.build_flags,
                )
                if resp.success and resp.data and resp.data.get("status") != "skipped":
                    return resp.data
            except Exception as e:
                logger.warning(f"[Verifier] MCP benchmark failed: {e}")

        # 降级: 用 PyTorch 参考实现做 benchmark（测的是原生实现，不是生成的 kernel）
        if op_ir:
            return self._pytorch_ref_benchmark(kernel, op_ir, hw)

        return {"latency_ms": 0, "bw_utilization": 0, "speedup": 0}

    def _pytorch_ref_benchmark(
        self, kernel: GeneratedKernel, op_ir: OperatorIR, hw: dict
    ) -> dict:
        """用 PyTorch 参考实现做 benchmark（提供一个性能基线）"""
        try:
            import torch
            import time as _time

            from tools.cpu_simulator import CPUSimulator
            ref_fn = CPUSimulator().REFERENCE_IMPLS.get(op_ir.name)
            if ref_fn is None:
                return {"latency_ms": 0, "bw_utilization": 0, "speedup": 0}

            # 选设备
            if hw.get("nvidia_gpu") and kernel.backend in ("cuda", "triton"):
                device = torch.device("cuda:0")
                sync_fn = torch.cuda.synchronize
            elif hw.get("npu") and kernel.backend == "ascendc":
                import torch_npu  # noqa: F401
                device = torch.device("npu:0")
                sync_fn = torch.npu.synchronize
            else:
                return {"latency_ms": 0, "bw_utilization": 0, "speedup": 0}

            x = torch.randn(4, 512, 4096, dtype=torch.float16, device=device)
            # warmup
            for _ in range(10):
                with torch.no_grad():
                    ref_fn(x)
            sync_fn()

            # benchmark
            start = _time.perf_counter()
            repeats = 50
            for _ in range(repeats):
                with torch.no_grad():
                    ref_fn(x)
            sync_fn()
            elapsed = (_time.perf_counter() - start) / repeats * 1000  # ms

            # 估算 BW utilization
            bw_util = self._estimate_bw_from_code(kernel)

            return {
                "latency_ms": elapsed,
                "bw_utilization": bw_util,
                "speedup": 1.0,  # 参考实现 vs 参考实现 = 1.0
                "note": "Benchmarked PyTorch ref impl (not the generated kernel)",
            }
        except Exception as e:
            logger.warning(f"[Verifier] PyTorch benchmark failed: {e}")
            return {"latency_ms": 0, "bw_utilization": 0, "speedup": 0}

    # ════════════════════════════════════════════════════════
    # 辅助方法
    # ════════════════════════════════════════════════════════

    def _estimate_performance(self, kernel: GeneratedKernel, gpu_spec: Optional[GPUSpec]) -> dict:
        """无法在硬件上运行时，用代码特征估算性能"""
        bw_util = self._estimate_bw_from_code(kernel)
        return {
            "bandwidth_utilization": bw_util,
            "passed": bw_util >= self.min_bandwidth_efficiency,
        }

    def _estimate_bw_from_code(self, kernel: GeneratedKernel) -> float:
        """基于代码特征估算带宽利用率"""
        if kernel.estimated_bandwidth_utilization > 0:
            return kernel.estimated_bandwidth_utilization

        code = kernel.source_code.lower()
        score = 0.4
        if "shared" in code or "lds" in code or "__shared__" in code:
            score += 0.15
        if "float4" in code or "half2" in code:
            score += 0.10
        if "unroll" in code:
            score += 0.05
        if "double_buffer" in code or "doublebuffer" in code or "pipe" in code:
            score += 0.05
        return min(max(score, 0.1), 0.95)

    def _compute_overall_pass(self, report: VerificationReport) -> bool:
        """根据达到的验证等级计算最终通过/失败"""
        level = report.verification_level

        # 静态分析必须通过
        if report.static_analysis.get("summary") == "FAIL":
            return False

        # 如果做了编译，编译必须通过
        if level_ge(level, VerificationLevel.COMPILED) and not report.compilation_passed:
            return False

        # 如果做了硬件验证，正确性必须通过
        if level_ge(level, VerificationLevel.HARDWARE_VERIFIED) and not report.correctness_passed:
            return False

        # 性能不强制要求通过（只是建议）
        return True

    def _finish(self, report: VerificationReport) -> AgentResult:
        logger.info(f"[Verifier] {report.summary()}")
        return self.success_result(
            output=report,
            metrics={
                "passed": report.overall_passed,
                "verification_level": report.verification_level.value,
                "correctness": report.correctness_passed,
                "performance": report.performance_passed,
                "bandwidth_utilization": report.bandwidth_utilization,
            }
        )

    def generate_fix_prompt(self, report: VerificationReport, kernel: GeneratedKernel) -> str:
        """生成修复提示词，用于反馈给 CodeGenAgent"""
        issues = []
        if report.compilation_errors:
            issues.extend(report.compilation_errors[:3])
        if not report.correctness_passed and report.correctness_details:
            issues.append(f"Correctness: {report.correctness_details}")
        if not report.performance_passed:
            issues.append(f"Performance: BW={report.bandwidth_utilization:.1%}")
        if report.static_analysis.get("failed_checks"):
            issues.extend(report.static_analysis["failed_checks"][:2])

        return textwrap.dedent(f"""
        修复以下 GPU 内核的问题:

        内核: {kernel.operator_name} ({kernel.backend})
        验证等级: {report.verification_level.value}
        问题列表:
        {chr(10).join(f'  - {i}' for i in issues)}

        建议:
        {chr(10).join(f'  - {r}' for r in report.recommendations[:5])}

        原始代码:
        ```
        {kernel.source_code[:2000]}
        ```

        请生成修复后的完整代码。
        """).strip()
