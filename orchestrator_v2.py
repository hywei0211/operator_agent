"""
Master Orchestrator V2 - 双路径主调度器
Path A：已知 GPU → 查仓库 → 直接训练
Path B：未知 GPU → 发现规格 → 生成算子 → Review Loop → 训练
"""
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from agents.base_agent import AgentContext, AgentResult, AgentStatus, BaseAgent
from agents.code_generator import CodeGenAgent
from agents.distribution import DistributionAgent
from agents.gpu_discovery import GPUDiscoveryAgent
from agents.optimizer import OptimizerAgent
from agents.review_loop import ReviewLoopAgent
from agents.runtime_monitor import RuntimeMonitorAgent
from agents.sdk_resolver import SDKResolverAgent
from agents.spec_analyzer import OperatorSpecAgent
from agents.tiling_agent import TilingAgent
from agents.training_analyst import TrainingAnalystAgent
from agents.training_executor import TrainingExecutorAgent
from mcp_servers.base_server import MCPClient
from mcp_servers.gpu_spec_server import GPUSpecMCPServer
from mcp_servers.operator_registry_server import OperatorRegistryMCPServer
from mcp_servers.remote_executor_server import RemoteExecutorMCPServer
from mcp_servers.sdk_docs_server import SDKDocsMCPServer
from models.operator_ir import ClusterConfig
from operators.registry import get_registry

logger = logging.getLogger(__name__)


class ExecutionPath(str, Enum):
    KNOWN_GPU = "known_gpu"       # Path A：已知 GPU
    UNKNOWN_GPU = "unknown_gpu"   # Path B：未知 GPU


@dataclass
class SystemConfig:
    llm_backend: str = "mock"           # openai / anthropic / mock
    llm_model: str = "gpt-4o"
    max_review_iterations: int = 5
    min_bandwidth_efficiency: float = 0.55
    dry_run_training: bool = True        # True = 只生成启动命令，不实际执行
    use_docker_compile: bool = False
    ssh_hosts: dict = field(default_factory=dict)
    parallel_operator_gen: bool = True   # 是否并行为多 GPU 生成算子


class MasterOrchestrator(BaseAgent):
    """
    双路径主调度器

    使用方式：
        orchestrator = MasterOrchestrator.create(config)
        result = await orchestrator.run(context,
            training_code="...",
            gpu_list=["h100_sxm5", "mi300x"],
        )
    """

    def __init__(self, mcp_client: MCPClient, llm_client=None, config: SystemConfig = None):
        super().__init__("MasterOrchestrator", llm_client, config.__dict__ if config else {})
        self.mcp = mcp_client
        self.sys_config = config or SystemConfig()

        # 初始化所有 Sub-Agent
        self._agents = self._init_agents()

    @classmethod
    def create(cls, config: SystemConfig = None) -> "MasterOrchestrator":
        """工厂方法：一键创建完整配置的 Orchestrator"""
        from tools.llm_client import create_llm_client
        cfg = config or SystemConfig()

        # 创建 LLM 客户端
        llm_client = create_llm_client(backend=cfg.llm_backend, model=cfg.llm_model)

        # 初始化 MCP Client，注册所有 Server
        mcp = MCPClient()
        mcp.register_server(GPUSpecMCPServer(llm_client=llm_client))
        mcp.register_server(SDKDocsMCPServer(llm_client=llm_client))
        mcp.register_server(OperatorRegistryMCPServer())
        mcp.register_server(RemoteExecutorMCPServer(
            ssh_hosts=cfg.ssh_hosts,
            use_docker=cfg.use_docker_compile,
        ))

        return cls(mcp_client=mcp, llm_client=llm_client, config=cfg)

    def _init_agents(self) -> dict:
        cfg_dict = self.sys_config.__dict__
        review_cfg = {
            "max_iterations": self.sys_config.max_review_iterations,
            "min_bandwidth_efficiency": self.sys_config.min_bandwidth_efficiency,
        }
        return {
            "gpu_discovery":     GPUDiscoveryAgent(self.mcp, self.llm_client),
            "sdk_resolver":      SDKResolverAgent(self.mcp, self.llm_client),
            "tiling":            TilingAgent(self.llm_client),
            "spec_analyzer":     OperatorSpecAgent(self.llm_client),
            "codegen":           CodeGenAgent(self.llm_client),
            "optimizer":         OptimizerAgent(self.llm_client),
            "review_loop":       ReviewLoopAgent(self.mcp, self.llm_client, review_cfg),
            "distribution":      DistributionAgent(self.llm_client),
            "training_analyst":  TrainingAnalystAgent(self.llm_client),
            "training_executor": TrainingExecutorAgent(self.llm_client, {"dry_run": self.sys_config.dry_run_training}),
            "runtime_monitor":   RuntimeMonitorAgent(self.llm_client),
        }

    def get_system_prompt(self) -> str:
        return "你是GPU算子生成和分布式训练调度的总指挥。"

    # ════════════════════════════════════════════════════════
    # 主执行入口
    # ════════════════════════════════════════════════════════

    async def run(self, context: AgentContext, **kwargs) -> AgentResult:
        self._start_timer()
        self.set_status(AgentStatus.RUNNING)

        training_code: str = kwargs.get("training_code", "")
        gpu_list: list[str] = kwargs.get("gpu_list", context.target_gpus)
        cluster_config: Optional[ClusterConfig] = kwargs.get("cluster_config")

        logger.info(f"[Master] Starting. GPUs={gpu_list}")
        logger.info(f"[Master] Training code length: {len(training_code)} chars")

        try:
            # ── Phase 0: 分析训练代码 ──────────────────────────────
            plan_result = await self._call("training_analyst", context,
                                          training_code=training_code)
            if not plan_result.success:
                return self.failure_result(f"Training analysis failed: {plan_result.error}")
            training_plan = plan_result.output
            logger.info(f"[Master] Operators needed: {training_plan.all_operators()}")

            # ── Phase 1: 判断执行路径 ──────────────────────────────
            path, known_gpus, unknown_gpus = self._classify_gpus(gpu_list)
            logger.info(f"[Master] Path={path.value}, known={known_gpus}, unknown={unknown_gpus}")

            # ── Phase 2A: 已知 GPU → 查仓库 ────────────────────────
            kernels_from_registry = {}
            if known_gpus:
                kernels_from_registry = self._load_from_registry(
                    training_plan.all_operators(), known_gpus
                )
                logger.info(f"[Master] Loaded {len(kernels_from_registry)} kernels from registry")

            # ── Phase 2B: 未知 GPU → 完整发现+生成流程 ─────────────
            kernels_generated = {}
            if unknown_gpus or self._needs_new_operators(training_plan, known_gpus):
                target_gpus = unknown_gpus or gpu_list

                # 2B-1: GPU 规格发现
                disc_result = await self._call("gpu_discovery", context,
                                              gpu_names=target_gpus)
                if not disc_result.success:
                    return self.failure_result(f"GPU discovery failed: {disc_result.error}")

                # 2B-2: SDK 解析
                await self._call("sdk_resolver", context)

                # 2B-3: 为每个算子生成内核（可并行）
                hardware_profiles = context.get_artifact("hardware_profiles") or {}
                kernels_generated = await self._generate_all_operators(
                    context, training_plan, hardware_profiles
                )

            # 合并所有内核
            all_kernels = {**kernels_from_registry, **kernels_generated}
            context.add_artifact("optimized_kernels", all_kernels)

            # ── Phase 3: 分布式策略 ────────────────────────────────
            cluster_cfg = cluster_config or self._build_cluster_config(gpu_list)
            await self._call("distribution", context,
                            kernels=all_kernels,
                            cluster_config=cluster_cfg,
                            operator_ir=context.get_artifact("operator_ir"))

            # ── Phase 4: 启动训练 ──────────────────────────────────
            exec_result = await self._call("training_executor", context,
                                          training_code=training_code,
                                          cluster_config=cluster_cfg,
                                          training_plan=training_plan)
            if not exec_result.success:
                return self.failure_result(f"Training launch failed: {exec_result.error}")

            # ── Phase 5: 启动监控 ──────────────────────────────────
            await self._call("runtime_monitor", context,
                            job=exec_result.output,
                            monitor_once=True)

            self.set_status(AgentStatus.COMPLETED)
            logger.info(f"[Master] Completed in {self._elapsed():.1f}s")

            return self.success_result(
                output={
                    "path": path.value,
                    "training_job": exec_result.output,
                    "kernels": {k: v.__class__.__name__ for k, v in all_kernels.items()},
                    "monitor": context.get_artifact("monitor_report"),
                    "elapsed_seconds": self._elapsed(),
                },
                metrics={
                    "path": path.value,
                    "known_gpus": len(known_gpus),
                    "unknown_gpus": len(unknown_gpus),
                    "operators_generated": len(kernels_generated),
                    "operators_from_registry": len(kernels_from_registry),
                }
            )

        except Exception as e:
            self.set_status(AgentStatus.FAILED)
            logger.exception(f"[Master] Failed: {e}")
            return self.failure_result(str(e))

    # ════════════════════════════════════════════════════════
    # 路径判断
    # ════════════════════════════════════════════════════════

    def _classify_gpus(
        self, gpu_list: list[str]
    ) -> tuple[ExecutionPath, list[str], list[str]]:
        """将 GPU 列表分为已知和未知两组"""
        from knowledge_base.hardware_specs.gpu_database import GPU_DATABASE
        known = [g for g in gpu_list if g.lower().replace(" ", "_") in GPU_DATABASE or
                 any(g.lower() in k for k in GPU_DATABASE)]
        unknown = [g for g in gpu_list if g not in known]
        path = ExecutionPath.UNKNOWN_GPU if unknown else ExecutionPath.KNOWN_GPU
        return path, known, unknown

    def _needs_new_operators(self, plan, known_gpus: list[str]) -> bool:
        """检查已知 GPU 是否有缺失的算子"""
        registry = get_registry()
        for gpu in known_gpus:
            for op in plan.all_operators():
                if not registry.lookup(op, gpu):
                    return True
        return False

    # ════════════════════════════════════════════════════════
    # 算子生成核心
    # ════════════════════════════════════════════════════════

    async def _generate_all_operators(
        self,
        context: AgentContext,
        training_plan,
        hardware_profiles: dict,
    ) -> dict:
        """为所有 GPU + 所有算子生成内核（支持并行）"""
        all_kernels = {}
        operators = training_plan.all_operators()

        if self.sys_config.parallel_operator_gen:
            # 并行：同时为所有 GPU 生成所有算子
            tasks = []
            for op_name in operators:
                for gpu_id, gpu_spec in hardware_profiles.items():
                    if gpu_spec:
                        tasks.append(self._generate_single_operator(
                            context, op_name, gpu_id, gpu_spec
                        ))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, dict):
                    all_kernels.update(result)
        else:
            # 串行：按算子优先级顺序生成
            for op_name in operators:
                for gpu_id, gpu_spec in hardware_profiles.items():
                    if gpu_spec:
                        result = await self._generate_single_operator(
                            context, op_name, gpu_id, gpu_spec
                        )
                        all_kernels.update(result)

        return all_kernels

    async def _generate_single_operator(
        self, context: AgentContext, op_name: str, gpu_id: str, gpu_spec
    ) -> dict:
        """
        为单个 (算子, GPU) 组合完整执行生成+Review流程
        """
        logger.info(f"[Master] Generating {op_name} for {gpu_id}")
        sub_ctx = AgentContext(operator_name=op_name, target_gpus=[gpu_id])

        # 1. 解析算子规格
        spec_result = await self._call("spec_analyzer", sub_ctx, request=op_name)
        if not spec_result.success:
            logger.warning(f"[Master] Spec analysis failed for {op_name}: {spec_result.error}")
            return {}
        op_ir = spec_result.output

        # 2. 获取 SDK 上下文
        sdk_contexts = context.get_artifact("sdk_contexts") or {}
        sdk_ctx = sdk_contexts.get(gpu_id)

        # 3. 计算 Tiling
        tiling_result = await self._call("tiling", sub_ctx,
                                        operator_ir=op_ir,
                                        gpu_spec=gpu_spec,
                                        sdk_context=sdk_ctx)
        tiling_config = tiling_result.output if tiling_result.success else None

        # 4. 生成初始代码
        gen_result = await self._call("codegen", sub_ctx,
                                     operator_ir=op_ir,
                                     gpu_spec=gpu_spec)
        if not gen_result.success:
            logger.warning(f"[Master] Code gen failed for {op_name}/{gpu_id}")
            return {}
        kernel = gen_result.output

        # 5. Review Loop（核心质量保障）
        review_result = await self._call("review_loop", sub_ctx,
                                        kernel=kernel,
                                        operator_ir=op_ir,
                                        gpu_spec=gpu_spec,
                                        tiling_config=tiling_config,
                                        sdk_context=sdk_ctx)

        if review_result.success and review_result.output:
            summary = review_result.output
            final_kernel = summary.final_kernel
            if final_kernel:
                return {f"{op_name}_{gpu_id}": final_kernel}

        return {}

    # ════════════════════════════════════════════════════════
    # 辅助方法
    # ════════════════════════════════════════════════════════

    def _load_from_registry(self, operators: list[str], gpus: list[str]) -> dict:
        """从算子仓库加载已有的算子"""
        registry = get_registry()
        loaded = {}
        for gpu in gpus:
            for op in operators:
                entry = registry.lookup(op, gpu) or registry.find_similar(op, gpu)
                if entry:
                    from models.operator_ir import GeneratedKernel
                    kernel = GeneratedKernel(
                        operator_name=entry.operator_name,
                        backend=entry.backend,
                        target_gpu=entry.gpu_model,
                        source_code=entry.source_code,
                        header_code=entry.header_code,
                        build_flags=entry.build_flags,
                        launch_config=entry.launch_config,
                        correctness_verified=entry.correctness_passed,
                        estimated_bandwidth_utilization=entry.bandwidth_utilization,
                    )
                    loaded[f"{op}_{gpu}"] = kernel
        return loaded

    def _build_cluster_config(self, gpu_list: list[str]) -> ClusterConfig:
        """根据 GPU 列表构建默认集群配置"""
        from collections import Counter
        counts = Counter(gpu_list)
        nodes = [{"gpu_model": g, "num_gpus": c} for g, c in counts.items()]
        gpu_groups = {g: [f"node_{i}"] for i, g in enumerate(counts)}
        return ClusterConfig(
            cluster_name="auto",
            nodes=nodes,
            gpu_groups=gpu_groups,
        )

    async def _call(self, agent_name: str, context: AgentContext, **kwargs) -> AgentResult:
        agent = self._agents.get(agent_name)
        if not agent:
            return AgentResult(success=False, agent_name=agent_name,
                              error=f"Agent {agent_name} not found")
        try:
            return await agent.run(context, **kwargs)
        except Exception as e:
            logger.error(f"[Master] Agent {agent_name} raised: {e}")
            return AgentResult(success=False, agent_name=agent_name, error=str(e))
