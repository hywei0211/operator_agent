"""
Orchestrator Agent - 主调度器
负责整个算子生成工作流的协调与调度
"""
import logging
from typing import Optional

from agents.base_agent import BaseAgent, AgentContext, AgentResult, AgentStatus
from models.operator_ir import OperatorIR, GeneratedKernel, ClusterConfig

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    """
    主调度器Agent

    工作流：
    1. 接收用户请求（算子描述 + 目标硬件集群）
    2. 调用 HardwareProfilerAgent 分析硬件规格
    3. 调用 OperatorSpecAgent 生成算子IR
    4. 为每种GPU类型并行调用 CodeGenAgent 生成内核
    5. 调用 OptimizerAgent 进行性能优化（迭代）
    6. 调用 VerifierAgent 验证正确性和性能
    7. 调用 DistributionAgent 生成分布式部署方案
    8. 返回最终结果
    """

    def __init__(self, llm_client=None, config: dict = None):
        super().__init__("OrchestratorAgent", llm_client, config)
        self._sub_agents: dict[str, BaseAgent] = {}

    def register_agent(self, agent: BaseAgent):
        """注册子Agent"""
        self._sub_agents[agent.name] = agent
        logger.info(f"[Orchestrator] Registered agent: {agent.name}")

    def get_system_prompt(self) -> str:
        return """你是一个GPU算子生成系统的总调度器。
你的职责是：
1. 理解用户需求，分解任务
2. 协调各专业Agent完成算子生成
3. 管理迭代优化循环
4. 确保最终输出满足正确性和性能要求

你需要生成一个工作流执行计划，指导各Agent按顺序完成任务。"""

    async def run(self, context: AgentContext, **kwargs) -> AgentResult:
        self._start_timer()
        self.set_status(AgentStatus.RUNNING)

        operator_request = kwargs.get("operator_request", "")
        cluster_config: Optional[ClusterConfig] = kwargs.get("cluster_config")

        logger.info(f"[Orchestrator] Starting workflow for: {operator_request}")
        logger.info(f"[Orchestrator] Target cluster: {cluster_config.cluster_name if cluster_config else 'default'}")

        try:
            # Step 1: 分析目标硬件
            hw_result = await self._run_sub_agent(
                "HardwareProfilerAgent", context,
                cluster_config=cluster_config
            )
            if not hw_result.success:
                return self.failure_result(f"Hardware profiling failed: {hw_result.error}")

            # Step 2: 解析算子规格，生成IR
            spec_result = await self._run_sub_agent(
                "OperatorSpecAgent", context,
                request=operator_request,
                hardware_profiles=hw_result.output
            )
            if not spec_result.success:
                return self.failure_result(f"Operator spec parsing failed: {spec_result.error}")

            operator_ir: OperatorIR = spec_result.output
            context.add_artifact("operator_ir", operator_ir)

            # Step 3: 为每种GPU类型生成内核代码
            generated_kernels = {}
            hardware_profiles = hw_result.output

            for gpu_type, gpu_spec in hardware_profiles.items():
                logger.info(f"[Orchestrator] Generating kernel for {gpu_type}")
                gen_result = await self._run_sub_agent(
                    "CodeGenAgent", context,
                    operator_ir=operator_ir,
                    gpu_spec=gpu_spec
                )
                if gen_result.success:
                    generated_kernels[gpu_type] = gen_result.output
                else:
                    logger.warning(f"[Orchestrator] Code gen failed for {gpu_type}: {gen_result.error}")

            if not generated_kernels:
                return self.failure_result("Failed to generate kernels for any GPU type")

            context.add_artifact("generated_kernels", generated_kernels)

            # Step 4: 优化迭代循环
            optimized_kernels = {}
            for gpu_type, kernel in generated_kernels.items():
                optimized_kernel = await self._optimize_kernel(context, kernel, hardware_profiles[gpu_type])
                optimized_kernels[gpu_type] = optimized_kernel

            context.add_artifact("optimized_kernels", optimized_kernels)

            # Step 5: 验证
            verification_results = {}
            all_verified = True
            for gpu_type, kernel in optimized_kernels.items():
                verify_result = await self._run_sub_agent(
                    "VerifierAgent", context,
                    kernel=kernel,
                    operator_ir=operator_ir,
                    gpu_spec=hardware_profiles[gpu_type]
                )
                verification_results[gpu_type] = verify_result
                if not verify_result.success:
                    logger.warning(f"[Orchestrator] Verification failed for {gpu_type}")
                    all_verified = False

            # Step 6: 分布式部署方案
            if cluster_config and cluster_config.is_heterogeneous():
                dist_result = await self._run_sub_agent(
                    "DistributionAgent", context,
                    kernels=optimized_kernels,
                    cluster_config=cluster_config,
                    operator_ir=operator_ir
                )
                context.add_artifact("distribution_plan", dist_result.output if dist_result.success else None)

            # 汇总结果
            final_output = {
                "operator_ir": operator_ir,
                "kernels": optimized_kernels,
                "verification": verification_results,
                "all_verified": all_verified,
                "distribution_plan": context.get_artifact("distribution_plan"),
                "metrics": self._collect_metrics(context),
            }

            self.set_status(AgentStatus.COMPLETED)
            logger.info(f"[Orchestrator] Workflow completed. Kernels generated: {list(optimized_kernels.keys())}")

            return self.success_result(
                output=final_output,
                metrics={"total_gpu_types": len(optimized_kernels), "all_verified": all_verified}
            )

        except Exception as e:
            self.set_status(AgentStatus.FAILED)
            logger.exception(f"[Orchestrator] Workflow failed: {e}")
            return self.failure_result(str(e))

    async def _optimize_kernel(self, context: AgentContext, kernel: GeneratedKernel, gpu_spec) -> GeneratedKernel:
        """对单个内核进行迭代优化"""
        current_kernel = kernel
        max_opt_iterations = self.config.get("max_opt_iterations", 3)

        for i in range(max_opt_iterations):
            opt_result = await self._run_sub_agent(
                "OptimizerAgent", context,
                kernel=current_kernel,
                gpu_spec=gpu_spec,
                iteration=i + 1
            )
            if not opt_result.success:
                logger.warning(f"[Orchestrator] Optimization iteration {i+1} failed, keeping previous version")
                break

            optimized = opt_result.output
            # 如果优化收益不大，提前终止
            if self._optimization_converged(current_kernel, optimized):
                logger.info(f"[Orchestrator] Optimization converged at iteration {i+1}")
                current_kernel = optimized
                break
            current_kernel = optimized

        return current_kernel

    def _optimization_converged(self, prev: GeneratedKernel, curr: GeneratedKernel) -> bool:
        """判断优化是否已收敛（改善幅度小于阈值）"""
        threshold = self.config.get("convergence_threshold", 0.02)
        prev_util = prev.estimated_bandwidth_utilization
        curr_util = curr.estimated_bandwidth_utilization
        if prev_util == 0:
            return False
        improvement = (curr_util - prev_util) / prev_util
        return improvement < threshold

    async def _run_sub_agent(self, agent_name: str, context: AgentContext, **kwargs) -> AgentResult:
        """调用子Agent并处理异常"""
        agent = self._sub_agents.get(agent_name)
        if agent is None:
            return AgentResult(
                success=False,
                agent_name=agent_name,
                error=f"Agent '{agent_name}' not registered"
            )
        try:
            self.set_status(AgentStatus.WAITING)
            result = await agent.run(context, **kwargs)
            self.set_status(AgentStatus.RUNNING)
            return result
        except Exception as e:
            logger.error(f"[Orchestrator] Sub-agent {agent_name} raised exception: {e}")
            return AgentResult(success=False, agent_name=agent_name, error=str(e))

    def _collect_metrics(self, context: AgentContext) -> dict:
        kernels = context.get_artifact("optimized_kernels", {})
        return {
            "total_gpu_types": len(kernels),
            "iterations": context.iteration_count,
            "kernel_names": list(kernels.keys()),
        }
