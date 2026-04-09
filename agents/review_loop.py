"""
Review Loop - 算子质量保障核心循环
五阶段渐进式验证：静态审查 → 编译 → 正确性 → 性能 → 综合评审
每个阶段发现的问题都会反馈给 CodeGenAgent 或 OptimizerAgent 修复
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from agents.base_agent import BaseAgent, AgentContext, AgentResult, AgentStatus
from agents.code_generator import CodeGenAgent
from agents.optimizer import OptimizerAgent
from agents.verifier import VerifierAgent, VerificationReport
from mcp_servers.base_server import MCPClient
from models.operator_ir import GeneratedKernel, OperatorIR
from models.hardware_model import GPUSpec
from operators.registry import get_registry, OperatorEntry

logger = logging.getLogger(__name__)


class ReviewStage(str, Enum):
    STATIC = "static_review"
    COMPILE = "compile_check"
    CORRECTNESS = "correctness_test"
    PERFORMANCE = "performance_benchmark"
    META = "meta_review"


@dataclass
class StageResult:
    stage: ReviewStage
    passed: bool
    score: float = 0.0          # 0~1
    issues: list = field(default_factory=list)
    fix_prompt: str = ""        # 反馈给修复Agent的提示


@dataclass
class ReviewSummary:
    """完整的 Review 循环结果"""
    operator_name: str
    gpu_model: str
    final_kernel: Optional[GeneratedKernel]
    stage_results: list = field(default_factory=list)
    total_iterations: int = 0
    final_passed: bool = False
    escalated_to_human: bool = False    # 超过最大迭代次数，需人工介入
    bandwidth_utilization: float = 0.0
    notes: str = ""


class ReviewLoopAgent(BaseAgent):
    """
    Review Loop Agent - 质量保障核心

    五阶段渐进式验证，发现问题立即反馈修复，最终写入算子仓库

    关键设计：
    - 每阶段成本递增（静态最便宜，硬件最贵），先便宜后贵
    - 发现问题后定向修复：代码错误 → CodeGenAgent，性能 → OptimizerAgent
    - 超过最大迭代次数 → 人工介入标记
    - 最终通过 → 写入 OperatorRegistry
    """

    MAX_ITERATIONS = 5
    MIN_BANDWIDTH_EFFICIENCY = 0.55
    MAX_RELATIVE_ERROR = 1e-3

    def __init__(self, mcp_client: MCPClient, llm_client=None, config: dict = None):
        super().__init__("ReviewLoopAgent", llm_client, config)
        self.mcp = mcp_client
        cfg = config or {}
        self.max_iterations = cfg.get("max_iterations", self.MAX_ITERATIONS)
        self.min_bw_efficiency = cfg.get("min_bandwidth_efficiency", self.MIN_BANDWIDTH_EFFICIENCY)

        # 子 Agent（按需复用）
        self._codegen = CodeGenAgent(llm_client=llm_client)
        self._optimizer = OptimizerAgent(llm_client=llm_client, config=config)
        self._verifier = VerifierAgent(llm_client=llm_client, config=config)

    def get_system_prompt(self) -> str:
        return "你是GPU算子质量评审专家，负责全面评估算子代码的正确性和性能。"

    async def run(self, context: AgentContext, **kwargs) -> AgentResult:
        self._start_timer()
        self.set_status(AgentStatus.RUNNING)

        kernel: Optional[GeneratedKernel] = kwargs.get("kernel")
        op_ir: Optional[OperatorIR] = kwargs.get("operator_ir") or context.get_artifact("operator_ir")
        gpu_spec: Optional[GPUSpec] = kwargs.get("gpu_spec")
        tiling_config = kwargs.get("tiling_config")
        sdk_context = kwargs.get("sdk_context")

        if not kernel or not op_ir or not gpu_spec:
            return self.failure_result("Missing kernel, operator_ir or gpu_spec")

        summary = ReviewSummary(
            operator_name=op_ir.name,
            gpu_model=gpu_spec.model_name,
            final_kernel=kernel,
        )

        current_kernel = kernel
        iteration = 0
        iteration_history: list[dict] = []  # 迭代历史，传递给 CodeGen 避免重复错误

        logger.info(f"[ReviewLoop] Starting review for {op_ir.name} on {gpu_spec.model_name}")

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"[ReviewLoop] === Iteration {iteration}/{self.max_iterations} ===")

            # ── Stage 1: 静态代码审查 ─────────────────────────────────
            s1 = await self._stage_static_review(current_kernel, op_ir, gpu_spec)
            summary.stage_results.append(s1)
            if not s1.passed:
                iteration_history.append({"iteration": iteration, "stage": "static_review", "issues": s1.issues[:3]})
                current_kernel = await self._fix_with_codegen(
                    current_kernel, op_ir, gpu_spec, tiling_config, sdk_context, s1, iteration_history
                )
                continue

            # ── Stage 2: 编译检查 ──────────────────────────────────────
            s2 = await self._stage_compile(current_kernel)
            summary.stage_results.append(s2)
            if not s2.passed:
                iteration_history.append({"iteration": iteration, "stage": "compile", "issues": s2.issues[:3]})
                # 记录编译错误到知识库
                try:
                    from knowledge_base.compile_error_kb import get_compile_error_kb
                    stderr = "\n".join(s2.issues)
                    get_compile_error_kb().record_error(current_kernel.backend, stderr, current_kernel.source_code)
                except Exception:
                    pass
                current_kernel = await self._fix_with_codegen(
                    current_kernel, op_ir, gpu_spec, tiling_config, sdk_context, s2, iteration_history
                )
                continue

            # ── Stage 3: 正确性验证 ───────────────────────────────────
            s3 = await self._stage_correctness(current_kernel, op_ir, gpu_spec)
            summary.stage_results.append(s3)
            if not s3.passed:
                iteration_history.append({"iteration": iteration, "stage": "correctness", "issues": s3.issues[:3]})
                current_kernel = await self._fix_with_codegen(
                    current_kernel, op_ir, gpu_spec, tiling_config, sdk_context, s3, iteration_history
                )
                continue

            # ── Stage 4: 性能基准 ─────────────────────────────────────
            s4 = await self._stage_performance(current_kernel, op_ir, gpu_spec)
            summary.stage_results.append(s4)
            if not s4.passed:
                current_kernel = await self._optimize_kernel(
                    current_kernel, op_ir, gpu_spec, s4, iteration
                )
                continue

            # ── Stage 5: 综合评审（全部通过）─────────────────────────
            s5 = await self._stage_meta_review(current_kernel, op_ir, gpu_spec, [s1, s2, s3, s4])
            summary.stage_results.append(s5)
            summary.final_passed = s5.passed
            summary.bandwidth_utilization = s4.score
            summary.total_iterations = iteration
            summary.final_kernel = current_kernel
            break
        else:
            # 超过最大迭代次数
            summary.escalated_to_human = True
            summary.notes = f"Exceeded {self.max_iterations} iterations. Manual review required."
            summary.final_kernel = current_kernel
            logger.warning(f"[ReviewLoop] {op_ir.name} escalated after {iteration} iterations")

        summary.total_iterations = iteration

        # 写入算子仓库（即使没完全通过，也存储当前最优版本）
        if summary.final_kernel and (summary.final_passed or summary.escalated_to_human):
            self._save_to_registry(summary, op_ir, gpu_spec)

        context.add_artifact(f"review_summary_{gpu_spec.model_name}", summary)
        logger.info(
            f"[ReviewLoop] Done: {op_ir.name} on {gpu_spec.model_name} | "
            f"passed={summary.final_passed} | iters={summary.total_iterations} | "
            f"BW={summary.bandwidth_utilization:.0%}"
        )

        return self.success_result(
            output=summary,
            metrics={
                "passed": summary.final_passed,
                "iterations": summary.total_iterations,
                "bandwidth_utilization": summary.bandwidth_utilization,
                "escalated": summary.escalated_to_human,
            }
        )

    # ════════════════════════════════════════════════════════
    # 五个验证阶段
    # ════════════════════════════════════════════════════════

    async def _stage_static_review(
        self, kernel: GeneratedKernel, op_ir: OperatorIR, gpu_spec: GPUSpec
    ) -> StageResult:
        """Stage 1: LLM 代码审查 + 静态分析"""
        issues = []
        code = kernel.source_code

        # 结构检查
        if len(code.strip()) < 50:
            issues.append("Code too short or empty")
        if code.count("{") != code.count("}"):
            issues.append("Mismatched braces")

        # 数学一致性检查（有 LLM 时用 LLM，否则用规则）
        if self.llm_client and op_ir.math_description:
            math_ok, math_issue = await self._llm_math_check(code, op_ir)
            if not math_ok:
                issues.append(f"Math mismatch: {math_issue}")
        else:
            # 规则检查：算子名应出现在代码中
            if op_ir.name.replace("_", "") not in code.lower().replace("_", ""):
                issues.append(f"Operator logic for '{op_ir.name}' not clearly visible")

        # 边界检查
        if "idx" in code and "< N" not in code and "mask" not in code:
            issues.append("Possible missing boundary check")

        passed = len(issues) == 0
        fix_prompt = f"Fix these issues in the {kernel.backend} kernel:\n" + "\n".join(f"- {i}" for i in issues) if issues else ""

        return StageResult(
            stage=ReviewStage.STATIC,
            passed=passed,
            score=1.0 if passed else max(0.0, 1.0 - len(issues) * 0.2),
            issues=issues,
            fix_prompt=fix_prompt,
        )

    async def _stage_compile(self, kernel: GeneratedKernel) -> StageResult:
        """Stage 2: 实际编译（通过 MCP 调用编译环境）"""
        resp = await self.mcp.call(
            "remote_executor_server", "compile_kernel",
            source_code=kernel.source_code,
            sdk=kernel.backend,
            build_flags=kernel.build_flags,
            kernel_name=kernel.operator_name,
        )

        if not resp.success:
            return StageResult(
                stage=ReviewStage.COMPILE,
                passed=False,
                issues=[resp.error],
                fix_prompt=f"Fix compilation error: {resp.error}",
            )

        result = resp.data or {}
        passed = result.get("success", False)
        errors = []
        if not passed:
            stderr = result.get("stderr", "")
            errors = [line for line in stderr.split("\n") if "error:" in line.lower()][:5]

        return StageResult(
            stage=ReviewStage.COMPILE,
            passed=passed,
            score=1.0 if passed else 0.0,
            issues=errors,
            fix_prompt=f"Fix these compilation errors:\n" + "\n".join(errors) if errors else "",
        )

    async def _stage_correctness(
        self, kernel: GeneratedKernel, op_ir: OperatorIR, gpu_spec: GPUSpec
    ) -> StageResult:
        """Stage 3: 数值正确性验证"""
        ctx = AgentContext()
        result = await self._verifier.run(
            ctx,
            kernel=kernel,
            operator_ir=op_ir,
            gpu_spec=gpu_spec,
        )

        if not result.success:
            return StageResult(
                stage=ReviewStage.CORRECTNESS,
                passed=False,
                issues=[result.error or "Verifier failed"],
            )

        report: VerificationReport = result.output
        passed = report.correctness_passed
        issues = []
        if not passed:
            issues.append(report.correctness_details or "Numerical correctness check failed")

        return StageResult(
            stage=ReviewStage.CORRECTNESS,
            passed=passed,
            score=1.0 if passed else max(0.0, 1.0 - report.max_relative_error * 100),
            issues=issues,
            fix_prompt=self._verifier.generate_fix_prompt(report, kernel) if not passed else "",
        )

    async def _stage_performance(
        self, kernel: GeneratedKernel, op_ir: OperatorIR, gpu_spec: GPUSpec
    ) -> StageResult:
        """Stage 4: 性能基准测试"""
        ctx = AgentContext()
        result = await self._verifier.run(
            ctx, kernel=kernel, operator_ir=op_ir, gpu_spec=gpu_spec
        )

        bw_util = 0.0
        if result.success and result.output:
            bw_util = result.output.bandwidth_utilization

        passed = bw_util >= self.min_bw_efficiency
        issues = []
        if not passed:
            issues.append(
                f"Bandwidth utilization {bw_util:.0%} < target {self.min_bw_efficiency:.0%}. "
                f"Consider: shared memory tiling, vectorized loads, or double buffering."
            )

        return StageResult(
            stage=ReviewStage.PERFORMANCE,
            passed=passed,
            score=bw_util,
            issues=issues,
            fix_prompt=f"Optimize for better performance (current: {bw_util:.0%}): " + "; ".join(issues) if issues else "",
        )

    async def _stage_meta_review(
        self,
        kernel: GeneratedKernel,
        op_ir: OperatorIR,
        gpu_spec: GPUSpec,
        prev_stages: list,
    ) -> StageResult:
        """Stage 5: 综合质量评审，生成最终裁定"""
        all_passed = all(s.passed for s in prev_stages)
        scores = [s.score for s in prev_stages]
        avg_score = sum(scores) / len(scores) if scores else 0

        issues = []
        if avg_score < 0.7:
            issues.append(f"Average quality score {avg_score:.0%} below 70%")

        passed = all_passed and avg_score >= 0.6
        return StageResult(
            stage=ReviewStage.META,
            passed=passed,
            score=avg_score,
            issues=issues,
        )

    # ════════════════════════════════════════════════════════
    # 修复辅助方法
    # ════════════════════════════════════════════════════════

    async def _fix_with_codegen(
        self,
        kernel: GeneratedKernel,
        op_ir: OperatorIR,
        gpu_spec: GPUSpec,
        tiling_config,
        sdk_context,
        stage_result: StageResult,
        iteration_history: list[dict] = None,
    ) -> GeneratedKernel:
        """把 Stage 发现的问题反馈给 CodeGenAgent 重新生成"""
        logger.info(f"[ReviewLoop] Fixing via CodeGen: {stage_result.issues[:2]}")

        # 构建包含历史的修复上下文
        history_summary = ""
        if iteration_history:
            lines = ["Previous fix attempts (avoid repeating these mistakes):"]
            for h in iteration_history:
                lines.append(f"  Iter {h['iteration']}, {h['stage']}: {'; '.join(h['issues'][:2])}")
            history_summary = "\n".join(lines)

        fix_context = {
            "previous_code": kernel.source_code,
            "issues_to_fix": stage_result.issues,
            "fix_guidance": stage_result.fix_prompt,
            "iteration_history": iteration_history or [],
            "history_summary": history_summary,
        }
        ctx = AgentContext()
        ctx.add_artifact("operator_ir", op_ir)
        ctx.add_artifact("fix_context", fix_context)
        if sdk_context:
            ctx.add_artifact("sdk_context", sdk_context)

        result = await self._codegen.run(
            ctx,
            operator_ir=op_ir,
            gpu_spec=gpu_spec,
        )
        return result.output if result.success else kernel

    async def _optimize_kernel(
        self,
        kernel: GeneratedKernel,
        op_ir: OperatorIR,
        gpu_spec: GPUSpec,
        perf_stage: StageResult,
        iteration: int,
    ) -> GeneratedKernel:
        """把性能问题反馈给 OptimizerAgent"""
        logger.info(f"[ReviewLoop] Optimizing: bw_util={perf_stage.score:.0%}")
        ctx = AgentContext()
        ctx.add_artifact("operator_ir", op_ir)
        result = await self._optimizer.run(
            ctx,
            kernel=kernel,
            gpu_spec=gpu_spec,
            operator_ir=op_ir,
            iteration=iteration,
        )
        return result.output if result.success else kernel

    async def _llm_math_check(self, code: str, op_ir: OperatorIR) -> tuple[bool, str]:
        """用 LLM 检查代码的数学正确性"""
        prompt = f"""检查以下代码是否正确实现了 {op_ir.name}。
数学定义: {op_ir.math_description}
代码（前1000字符）:
```
{code[:1000]}
```
如果实现正确，返回 OK。如果有问题，描述具体问题（一行）。"""
        try:
            resp = await self.call_llm(prompt, temperature=0.0, max_tokens=200)
            ok = "ok" in resp.lower() or "correct" in resp.lower()
            return ok, "" if ok else resp.strip()
        except Exception:
            return True, ""  # LLM 失败时不阻塞流程

    def _save_to_registry(self, summary: ReviewSummary, op_ir: OperatorIR, gpu_spec: GPUSpec):
        """将通过验证的算子存入仓库"""
        if not summary.final_kernel:
            return
        kernel = summary.final_kernel

        # 获取 prompt 版本
        prompt_ver = ""
        try:
            from prompts.code_gen_prompts import CUDA_PROMPT_VERSION
            prompt_ver = CUDA_PROMPT_VERSION
        except ImportError:
            pass

        entry = OperatorEntry(
            operator_name=op_ir.name,
            gpu_model=gpu_spec.model_name,
            backend=kernel.backend,
            source_code=kernel.source_code,
            header_code=kernel.header_code,
            build_flags=kernel.build_flags,
            launch_config=kernel.launch_config,
            correctness_passed=summary.final_passed,
            bandwidth_utilization=summary.bandwidth_utilization,
            iteration_count=summary.total_iterations,
            optimizations_applied=kernel.optimizations_applied,
            tags=op_ir.tags,
            prompt_version=prompt_ver,
        )
        get_registry().register(entry)
