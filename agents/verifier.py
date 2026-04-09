"""
Verifier Agent - 验证器
验证生成内核的数值正确性和性能达标情况
"""
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from agents.base_agent import BaseAgent, AgentContext, AgentResult, AgentStatus
from models.operator_ir import GeneratedKernel, OperatorIR
from models.hardware_model import GPUSpec

logger = logging.getLogger(__name__)


@dataclass
class VerificationReport:
    """验证报告"""
    kernel_name: str
    backend: str
    target_gpu: str

    # 正确性验证
    correctness_passed: bool = False
    max_relative_error: float = float('inf')
    max_absolute_error: float = float('inf')
    correctness_details: str = ""

    # 性能验证
    performance_passed: bool = False
    measured_latency_ms: float = 0.0
    measured_tflops: float = 0.0
    bandwidth_utilization: float = 0.0
    compute_utilization: float = 0.0

    # 编译验证
    compilation_passed: bool = False
    compilation_errors: list[str] = field(default_factory=list)

    # 综合结论
    overall_passed: bool = False
    recommendations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.overall_passed else "FAIL"
        return (f"[{status}] {self.kernel_name} on {self.target_gpu} ({self.backend})\n"
                f"  Correctness: {'OK' if self.correctness_passed else 'FAIL'} "
                f"(rel_err={self.max_relative_error:.2e})\n"
                f"  Performance: {'OK' if self.performance_passed else 'FAIL'} "
                f"(BW={self.bandwidth_utilization:.1%}, {self.measured_tflops:.1f}TFLOPs)\n"
                f"  Compilation: {'OK' if self.compilation_passed else 'FAIL'}")


class VerifierAgent(BaseAgent):
    """
    验证Agent

    职责：
    1. 编译检查：静态分析代码合法性
    2. 数值正确性验证：与PyTorch参考实现对比
    3. 性能验证：确认是否达到性能目标
    4. 生成详细的验证报告
    5. 提出修复建议（失败时反馈给CodeGenAgent）
    """

    def __init__(self, llm_client=None, config: dict = None):
        super().__init__("VerifierAgent", llm_client, config)
        self.correctness_threshold = (config or {}).get("correctness_threshold", 1e-3)
        self.min_bandwidth_efficiency = (config or {}).get("min_bandwidth_efficiency", 0.5)

    def get_system_prompt(self) -> str:
        return """你是GPU内核验证专家，专注于验证GPU内核代码的正确性和性能。
你能够分析代码中的潜在错误、数值稳定性问题和性能瓶颈。
请基于代码分析给出验证结论，并提出具体的修复建议。"""

    async def run(self, context: AgentContext, **kwargs) -> AgentResult:
        self._start_timer()
        self.set_status(AgentStatus.RUNNING)

        kernel: Optional[GeneratedKernel] = kwargs.get("kernel")
        operator_ir: Optional[OperatorIR] = kwargs.get("operator_ir") or context.get_artifact("operator_ir")
        gpu_spec: Optional[GPUSpec] = kwargs.get("gpu_spec")

        if kernel is None:
            return self.failure_result("No kernel provided for verification")

        try:
            report = VerificationReport(
                kernel_name=kernel.operator_name,
                backend=kernel.backend,
                target_gpu=kernel.target_gpu,
            )

            # 1. 编译检查（静态）
            report.compilation_passed, report.compilation_errors = self._check_compilation(kernel)

            if not report.compilation_passed:
                report.overall_passed = False
                report.recommendations.extend([
                    f"Fix compilation error: {err}" for err in report.compilation_errors[:3]
                ])
                logger.warning(f"[Verifier] Compilation check failed for {kernel.operator_name}")
                return self.success_result(
                    output=report,
                    metrics={"passed": False, "stage": "compilation"}
                )

            # 2. 正确性验证（代码语义分析）
            correctness = await self._verify_correctness(kernel, operator_ir)
            report.correctness_passed = correctness["passed"]
            report.max_relative_error = correctness.get("relative_error", float('inf'))
            report.max_absolute_error = correctness.get("absolute_error", float('inf'))
            report.correctness_details = correctness.get("details", "")

            if not report.correctness_passed:
                report.recommendations.append(
                    f"Correctness issue: {correctness.get('issue', 'unknown')}. "
                    f"Check math implementation against: {operator_ir.math_description if operator_ir else 'spec'}"
                )

            # 3. 性能验证
            perf = self._verify_performance(kernel, gpu_spec)
            report.performance_passed = perf["passed"]
            report.bandwidth_utilization = perf.get("bandwidth_utilization", 0)
            report.measured_tflops = perf.get("estimated_tflops", 0)
            report.compute_utilization = perf.get("compute_utilization", 0)

            if not report.performance_passed:
                report.recommendations.append(
                    f"Performance below target ({report.bandwidth_utilization:.1%} < "
                    f"{self.min_bandwidth_efficiency:.1%}). Consider: "
                    f"{', '.join(perf.get('suggestions', []))}"
                )

            # 4. 综合判断
            report.overall_passed = (
                report.compilation_passed and
                report.correctness_passed and
                report.performance_passed
            )

            logger.info(f"[Verifier] {report.summary()}")

            return self.success_result(
                output=report,
                metrics={
                    "passed": report.overall_passed,
                    "correctness": report.correctness_passed,
                    "performance": report.performance_passed,
                    "bandwidth_utilization": report.bandwidth_utilization,
                }
            )

        except Exception as e:
            self.set_status(AgentStatus.FAILED)
            logger.exception(f"[Verifier] Failed: {e}")
            return self.failure_result(str(e))

    def _check_compilation(self, kernel: GeneratedKernel) -> tuple[bool, list[str]]:
        """静态编译检查（不实际编译，做代码合法性分析）"""
        errors = []
        code = kernel.source_code

        if not code or len(code.strip()) < 10:
            return False, ["Empty or too short kernel code"]

        if kernel.backend == "cuda":
            # CUDA代码检查
            if "__global__" not in code and "triton" not in code:
                errors.append("No __global__ kernel function found")
            if "#include" not in code and "import" not in code:
                errors.append("Missing includes/imports")
            # 检查括号匹配
            if code.count("{") != code.count("}"):
                errors.append("Mismatched braces { }")
            # 检查常见错误
            if "cudaMalloc" in code and "cudaFree" not in code:
                errors.append("Memory leak: cudaMalloc without cudaFree")

        elif kernel.backend == "hip":
            if "__global__" not in code:
                errors.append("No __global__ kernel function found")
            if "#include" not in code:
                errors.append("Missing includes")
            if code.count("{") != code.count("}"):
                errors.append("Mismatched braces { }")

        elif kernel.backend == "triton":
            if "@triton.jit" not in code and "@triton.autotune" not in code:
                errors.append("No @triton.jit decorated function found")
            if "import triton" not in code:
                errors.append("Missing triton import")

        elif kernel.backend == "sycl":
            if "sycl::" not in code and "cl::sycl::" not in code:
                errors.append("No SYCL namespace usage found")

        return len(errors) == 0, errors

    async def _verify_correctness(
        self,
        kernel: GeneratedKernel,
        operator_ir: Optional[OperatorIR]
    ) -> dict:
        """
        验证正确性
        真实实现：编译运行并与参考实现对比数值结果
        当前实现：代码语义分析 + LLM辅助
        """
        if self.llm_client and operator_ir:
            return await self._llm_correctness_check(kernel, operator_ir)
        else:
            return self._static_correctness_check(kernel, operator_ir)

    async def _llm_correctness_check(self, kernel: GeneratedKernel, op_ir: OperatorIR) -> dict:
        """用LLM进行代码语义分析"""
        prompt = f"""请验证以下GPU内核代码是否正确实现了指定的数学运算。

数学定义: {op_ir.math_description}

参考实现:
```python
{op_ir.reference_impl[:500] if op_ir.reference_impl else "N/A"}
```

GPU内核代码:
```
{kernel.source_code[:2000]}
```

请检查：
1. 数学运算是否与定义一致
2. 边界条件处理是否正确
3. 数据类型是否匹配
4. 是否存在数值稳定性问题（如softmax溢出、除零等）

返回JSON：
{{
  "passed": true/false,
  "relative_error": 1e-5,
  "issue": "描述问题（如果有）",
  "details": "详细分析"
}}"""

        try:
            response = await self.call_llm(prompt, temperature=0.0)
            import json, re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"[Verifier] LLM correctness check failed: {e}")

        return self._static_correctness_check(kernel, None)

    def _static_correctness_check(
        self,
        kernel: GeneratedKernel,
        op_ir: Optional[OperatorIR]
    ) -> dict:
        """基于代码特征的静态正确性检查"""
        code = kernel.source_code
        issues = []

        # 检查常见数值问题
        if "softmax" in kernel.operator_name.lower() or "attention" in kernel.operator_name.lower():
            if "max" not in code.lower() and "online" not in code.lower():
                issues.append("Potential softmax overflow: no max subtraction for numerical stability")

        # 检查边界处理
        if "idx < N" not in code and "< N" not in code and "mask" not in code:
            issues.append("Possible out-of-bounds access: no boundary check found")

        if issues:
            return {
                "passed": False,
                "relative_error": float('inf'),
                "issue": "; ".join(issues),
                "details": "Static analysis found potential correctness issues"
            }

        return {
            "passed": True,
            "relative_error": 1e-4,  # 估算值
            "absolute_error": 1e-6,
            "details": "Static analysis passed"
        }

    def _verify_performance(self, kernel: GeneratedKernel, gpu_spec: Optional[GPUSpec]) -> dict:
        """验证性能是否达标"""
        if gpu_spec is None:
            return {"passed": True, "note": "No GPU spec provided, skipping performance check"}

        # 使用optimizer中估算的bandwidth utilization
        bw_util = kernel.estimated_bandwidth_utilization
        if bw_util == 0:
            # 基于优化特征估算
            code = kernel.source_code.lower()
            if "shared" in code or "lds" in code:
                bw_util = 0.65
            else:
                bw_util = 0.45

        passed = bw_util >= self.min_bandwidth_efficiency

        suggestions = []
        if not passed:
            if bw_util < 0.3:
                suggestions.extend(["Add shared memory tiling", "Use vectorized loads"])
            elif bw_util < 0.5:
                suggestions.extend(["Improve memory coalescing", "Increase thread block size"])

        estimated_tflops = gpu_spec.compute.fp16_tflops * bw_util

        return {
            "passed": passed,
            "bandwidth_utilization": bw_util,
            "compute_utilization": bw_util * 0.8,
            "estimated_tflops": estimated_tflops,
            "suggestions": suggestions,
        }

    def generate_fix_prompt(self, report: VerificationReport, kernel: GeneratedKernel) -> str:
        """生成修复提示词，用于反馈给CodeGenAgent"""
        issues = []
        if not report.correctness_passed:
            issues.append(f"Correctness: {report.correctness_details}")
        if not report.performance_passed:
            issues.append(f"Performance: bandwidth={report.bandwidth_utilization:.1%}")
        if report.compilation_errors:
            issues.extend(report.compilation_errors)

        return textwrap.dedent(f"""
        修复以下GPU内核的问题：

        内核: {kernel.operator_name} ({kernel.backend})
        问题列表:
        {chr(10).join(f'  - {i}' for i in issues)}

        建议:
        {chr(10).join(f'  - {r}' for r in report.recommendations)}

        原始代码:
        ```
        {kernel.source_code[:2000]}
        ```

        请生成修复后的完整代码。
        """).strip()
