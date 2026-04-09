"""
Distribution Agent - 分布式协调器
负责为异构GPU集群生成分布式训练的算子部署方案
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

from agents.base_agent import BaseAgent, AgentContext, AgentResult, AgentStatus
from models.operator_ir import GeneratedKernel, OperatorIR, ClusterConfig, ParallelStrategy

logger = logging.getLogger(__name__)


@dataclass
class DevicePlacement:
    """单个设备的算子部署方案"""
    device_id: str
    gpu_model: str
    kernel: GeneratedKernel
    shard_spec: dict = field(default_factory=dict)   # 张量分片规格
    rank: int = 0
    local_rank: int = 0


@dataclass
class DistributionPlan:
    """完整的分布式部署方案"""
    cluster_name: str
    operator_name: str
    parallel_strategy: ParallelStrategy
    total_gpus: int

    # 设备部署列表
    device_placements: list[DevicePlacement] = field(default_factory=list)

    # 通信方案
    communication_backend: str = "nccl"         # nccl / rccl / oneccl / mpi
    communication_pattern: str = "allreduce"    # allreduce / allgather / reducescatter / p2p
    communication_calls: list[dict] = field(default_factory=list)

    # 异构处理
    is_heterogeneous: bool = False
    load_balance_weights: dict[str, float] = field(default_factory=dict)  # GPU -> 负载权重
    synchronization_points: list[str] = field(default_factory=list)

    # 性能预测
    estimated_throughput_tflops: float = 0.0
    communication_overhead_pct: float = 0.0
    load_imbalance_pct: float = 0.0

    def summary(self) -> str:
        return (
            f"DistributionPlan: {self.operator_name} on {self.cluster_name}\n"
            f"  Strategy: {self.parallel_strategy.value}\n"
            f"  GPUs: {self.total_gpus} (heterogeneous={self.is_heterogeneous})\n"
            f"  Backend: {self.communication_backend}\n"
            f"  Est. throughput: {self.estimated_throughput_tflops:.1f} TFLOPs\n"
            f"  Comm overhead: {self.communication_overhead_pct:.1f}%"
        )


class DistributionAgent(BaseAgent):
    """
    分布式协调Agent

    职责：
    1. 分析集群拓扑和算子并行化需求
    2. 为异构GPU集群制定负载均衡方案
    3. 确定通信后端和通信模式
    4. 生成每个设备的算子分片方案
    5. 处理不同性能GPU之间的负载均衡
    6. 生成分布式初始化和通信代码

    关键挑战：
    - 异构GPU性能差异大 → 需要动态负载均衡
    - 不同后端的通信库不兼容 → 需要统一抽象层
    - 跨节点延迟高 → 需要通信-计算重叠
    """

    def __init__(self, llm_client=None, config: dict = None):
        super().__init__("DistributionAgent", llm_client, config)

    def get_system_prompt(self) -> str:
        return """你是分布式GPU计算专家，精通分布式深度学习训练。
你熟悉NCCL、RCCL、oneCCL等通信库，以及各种并行策略（数据并行、张量并行、流水线并行）。
你能够为异构GPU集群设计最优的算子分布方案，平衡计算负载和通信开销。"""

    async def run(self, context: AgentContext, **kwargs) -> AgentResult:
        self._start_timer()
        self.set_status(AgentStatus.RUNNING)

        kernels: dict[str, GeneratedKernel] = kwargs.get("kernels", {})
        cluster_config: Optional[ClusterConfig] = kwargs.get("cluster_config")
        operator_ir: Optional[OperatorIR] = kwargs.get("operator_ir") or context.get_artifact("operator_ir")
        hardware_profiles = context.get_artifact("hardware_profiles", {})

        if not kernels or cluster_config is None:
            return self.failure_result("Missing kernels or cluster_config")

        try:
            # 1. 分析集群拓扑
            topology = self._analyze_topology(cluster_config, hardware_profiles)
            logger.info(f"[Distribution] Cluster topology: {topology}")

            # 2. 确定并行策略
            strategy = self._select_parallel_strategy(operator_ir, cluster_config, topology)
            logger.info(f"[Distribution] Parallel strategy: {strategy.value}")

            # 3. 计算负载均衡权重
            weights = self._compute_load_balance(hardware_profiles, cluster_config)
            logger.info(f"[Distribution] Load balance weights: {weights}")

            # 4. 确定通信后端
            comm_backend = self._select_communication_backend(hardware_profiles)
            logger.info(f"[Distribution] Communication backend: {comm_backend}")

            # 5. 生成分片方案
            device_placements = self._generate_device_placements(
                kernels, cluster_config, hardware_profiles, strategy, weights
            )

            # 6. 设计通信模式
            comm_pattern = self._design_communication_pattern(strategy, operator_ir)

            # 7. 生成通信代码
            comm_calls = await self._generate_communication_code(
                strategy, cluster_config, comm_backend, comm_pattern, operator_ir
            )

            # 8. 估算整体性能
            perf_estimate = self._estimate_distributed_performance(
                hardware_profiles, weights, cluster_config, comm_pattern
            )

            plan = DistributionPlan(
                cluster_name=cluster_config.cluster_name,
                operator_name=list(kernels.values())[0].operator_name if kernels else "unknown",
                parallel_strategy=strategy,
                total_gpus=cluster_config.total_gpus(),
                device_placements=device_placements,
                communication_backend=comm_backend,
                communication_pattern=comm_pattern,
                communication_calls=comm_calls,
                is_heterogeneous=cluster_config.is_heterogeneous(),
                load_balance_weights=weights,
                estimated_throughput_tflops=perf_estimate["throughput"],
                communication_overhead_pct=perf_estimate["comm_overhead_pct"],
                load_imbalance_pct=perf_estimate["imbalance_pct"],
            )

            logger.info(f"[Distribution] {plan.summary()}")

            return self.success_result(
                output=plan,
                metrics={
                    "total_gpus": plan.total_gpus,
                    "strategy": strategy.value,
                    "comm_backend": comm_backend,
                    "heterogeneous": plan.is_heterogeneous,
                    "estimated_throughput": plan.estimated_throughput_tflops,
                }
            )

        except Exception as e:
            self.set_status(AgentStatus.FAILED)
            logger.exception(f"[Distribution] Failed: {e}")
            return self.failure_result(str(e))

    def _analyze_topology(self, cluster_config: ClusterConfig, hardware_profiles: dict) -> dict:
        """分析集群拓扑结构"""
        vendors = set()
        for gpu_model in cluster_config.gpu_groups:
            spec = hardware_profiles.get(gpu_model)
            if spec:
                vendors.add(spec.vendor.value)

        num_nodes = len(cluster_config.nodes)
        total_gpus = cluster_config.total_gpus()
        gpus_per_node = total_gpus // max(num_nodes, 1)

        return {
            "num_nodes": num_nodes,
            "total_gpus": total_gpus,
            "gpus_per_node": gpus_per_node,
            "vendors": list(vendors),
            "is_multi_vendor": len(vendors) > 1,
            "is_multi_node": num_nodes > 1,
            "intra_node_bw": cluster_config.intra_node_bandwidth_gbps,
            "inter_node_bw": cluster_config.inter_node_bandwidth_gbps,
        }

    def _select_parallel_strategy(
        self,
        operator_ir: Optional[OperatorIR],
        cluster_config: ClusterConfig,
        topology: dict
    ) -> ParallelStrategy:
        """选择最优并行策略"""
        # 如果算子IR已经指定了并行策略
        if operator_ir and operator_ir.parallel_strategy != ParallelStrategy.NONE:
            return operator_ir.parallel_strategy

        # 根据集群规模和算子类型推断
        total_gpus = topology["total_gpus"]

        if operator_ir and "moe" in operator_ir.tags:
            return ParallelStrategy.EXPERT_PARALLEL

        if total_gpus <= 8 and not topology["is_multi_node"]:
            return ParallelStrategy.TENSOR_PARALLEL

        if topology["is_multi_node"] and total_gpus > 64:
            return ParallelStrategy.DATA_PARALLEL

        return ParallelStrategy.TENSOR_PARALLEL

    def _compute_load_balance(
        self,
        hardware_profiles: dict,
        cluster_config: ClusterConfig
    ) -> dict[str, float]:
        """
        计算异构GPU的负载均衡权重
        基于各GPU的峰值算力比例分配工作量
        """
        weights = {}
        total_fp16 = 0.0

        for gpu_model, node_list in cluster_config.gpu_groups.items():
            spec = hardware_profiles.get(gpu_model)
            if spec:
                gpu_fp16 = spec.compute.fp16_tflops * len(node_list)
                weights[gpu_model] = gpu_fp16
                total_fp16 += gpu_fp16

        # 归一化
        if total_fp16 > 0:
            weights = {k: v / total_fp16 for k, v in weights.items()}

        return weights

    def _select_communication_backend(self, hardware_profiles: dict) -> str:
        """
        选择通信后端
        异构集群需要支持多种后端，或使用统一抽象层
        """
        from models.hardware_model import GPUVendor
        vendors = {spec.vendor for spec in hardware_profiles.values()}

        if len(vendors) == 1:
            vendor = next(iter(vendors))
            if vendor == GPUVendor.NVIDIA:
                return "nccl"
            elif vendor == GPUVendor.AMD:
                return "rccl"
            elif vendor == GPUVendor.INTEL:
                return "oneccl"

        # 异构集群：使用MPI或UCC（统一通信层）
        return "ucc"  # UCX Collective Communication (支持异构)

    def _generate_device_placements(
        self,
        kernels: dict[str, GeneratedKernel],
        cluster_config: ClusterConfig,
        hardware_profiles: dict,
        strategy: ParallelStrategy,
        weights: dict[str, float]
    ) -> list[DevicePlacement]:
        """生成每个设备的算子部署方案"""
        placements = []
        global_rank = 0

        for node_idx, node_info in enumerate(cluster_config.nodes):
            gpu_model = node_info.get("gpu_model", list(cluster_config.gpu_groups.keys())[0])
            num_gpus = node_info.get("num_gpus", 1)
            kernel = kernels.get(gpu_model) or list(kernels.values())[0]  # 兜底

            for local_rank in range(num_gpus):
                shard_spec = self._compute_shard_spec(
                    strategy, global_rank, cluster_config.total_gpus(),
                    weights.get(gpu_model, 1.0 / cluster_config.total_gpus())
                )
                placements.append(DevicePlacement(
                    device_id=f"node{node_idx}_gpu{local_rank}",
                    gpu_model=gpu_model,
                    kernel=kernel,
                    shard_spec=shard_spec,
                    rank=global_rank,
                    local_rank=local_rank,
                ))
                global_rank += 1

        return placements

    def _compute_shard_spec(
        self,
        strategy: ParallelStrategy,
        rank: int,
        world_size: int,
        weight: float
    ) -> dict:
        """计算张量分片规格"""
        if strategy == ParallelStrategy.TENSOR_PARALLEL:
            return {
                "type": "column_parallel",
                "rank": rank,
                "world_size": world_size,
                "partition_dim": -1,  # 最后一维分片
                "weight": weight,
            }
        elif strategy == ParallelStrategy.DATA_PARALLEL:
            return {
                "type": "replicated",
                "rank": rank,
                "world_size": world_size,
                "batch_shard": rank,
            }
        elif strategy == ParallelStrategy.SEQUENCE_PARALLEL:
            return {
                "type": "sequence_shard",
                "rank": rank,
                "world_size": world_size,
                "partition_dim": 1,  # sequence维度分片
            }
        return {"type": "full", "rank": rank}

    def _design_communication_pattern(
        self,
        strategy: ParallelStrategy,
        operator_ir: Optional[OperatorIR]
    ) -> str:
        """设计通信模式"""
        pattern_map = {
            ParallelStrategy.DATA_PARALLEL: "allreduce",
            ParallelStrategy.TENSOR_PARALLEL: "allreduce",
            ParallelStrategy.SEQUENCE_PARALLEL: "allgather+reducescatter",
            ParallelStrategy.PIPELINE_PARALLEL: "p2p",
            ParallelStrategy.EXPERT_PARALLEL: "alltoall",
        }
        return pattern_map.get(strategy, "allreduce")

    async def _generate_communication_code(
        self,
        strategy: ParallelStrategy,
        cluster_config: ClusterConfig,
        comm_backend: str,
        comm_pattern: str,
        operator_ir: Optional[OperatorIR]
    ) -> list[dict]:
        """生成通信代码片段"""
        comm_calls = []

        if comm_pattern == "allreduce":
            comm_calls.append({
                "type": "allreduce",
                "code_snippet": self._allreduce_snippet(comm_backend),
                "when": "after_backward",
                "tensor": "grad",
            })
        elif comm_pattern == "allgather+reducescatter":
            comm_calls.append({
                "type": "allgather",
                "code_snippet": self._allgather_snippet(comm_backend),
                "when": "before_forward",
                "tensor": "input",
            })
            comm_calls.append({
                "type": "reduce_scatter",
                "code_snippet": self._reducescatter_snippet(comm_backend),
                "when": "after_forward",
                "tensor": "output",
            })
        elif comm_pattern == "alltoall":
            comm_calls.append({
                "type": "alltoall",
                "code_snippet": self._alltoall_snippet(comm_backend),
                "when": "dispatch_and_combine",
                "tensor": "expert_input",
            })

        return comm_calls

    def _estimate_distributed_performance(
        self,
        hardware_profiles: dict,
        weights: dict,
        cluster_config: ClusterConfig,
        comm_pattern: str
    ) -> dict:
        """估算分布式性能"""
        total_fp16 = sum(
            spec.compute.fp16_tflops
            for spec in hardware_profiles.values()
        )

        # 通信开销估算（基于通信量/带宽）
        inter_node_bw = cluster_config.inter_node_bandwidth_gbps
        comm_overhead = 5.0 if inter_node_bw >= 400 else 15.0  # %
        if comm_pattern in ("alltoall",):
            comm_overhead *= 2

        # 负载不均衡损耗
        if weights:
            max_w = max(weights.values())
            min_w = min(weights.values())
            imbalance = (max_w - min_w) / max_w * 100 if max_w > 0 else 0
        else:
            imbalance = 0

        effective_tflops = total_fp16 * (1 - comm_overhead / 100) * (1 - imbalance / 100)

        return {
            "throughput": effective_tflops,
            "comm_overhead_pct": comm_overhead,
            "imbalance_pct": imbalance,
        }

    # ---- 通信代码片段生成 ----

    def _allreduce_snippet(self, backend: str) -> str:
        if backend == "nccl":
            return """// NCCL AllReduce
ncclAllReduce(grad_ptr, grad_ptr, count, ncclFloat16, ncclSum, comm, stream);"""
        elif backend == "rccl":
            return """// RCCL AllReduce
rcclAllReduce(grad_ptr, grad_ptr, count, rcclFloat16, rcclSum, comm, stream);"""
        elif backend == "ucc":
            return """# UCC AllReduce (heterogeneous)
dist.all_reduce(grad_tensor, op=dist.ReduceOp.SUM)"""
        return f"// {backend} AllReduce"

    def _allgather_snippet(self, backend: str) -> str:
        if backend in ("nccl", "rccl"):
            return """// AllGather for Sequence Parallel
ncclAllGather(input_ptr, output_ptr, count_per_rank, ncclFloat16, comm, stream);"""
        return """# AllGather
dist.all_gather(output_list, input_tensor)"""

    def _reducescatter_snippet(self, backend: str) -> str:
        if backend in ("nccl", "rccl"):
            return """// ReduceScatter for Sequence Parallel
ncclReduceScatter(input_ptr, output_ptr, count_per_rank, ncclFloat16, ncclSum, comm, stream);"""
        return """# ReduceScatter
dist.reduce_scatter(output_tensor, input_list)"""

    def _alltoall_snippet(self, backend: str) -> str:
        return """# AllToAll for MoE Expert Parallel
dist.all_to_all(output_splits, input_splits, output_tensor, input_tensor)"""
