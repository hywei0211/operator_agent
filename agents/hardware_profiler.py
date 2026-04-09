"""
Hardware Profiler Agent - 硬件分析器
负责分析目标GPU集群的硬件规格，为代码生成提供硬件上下文
"""
import logging
from typing import Optional

from agents.base_agent import BaseAgent, AgentContext, AgentResult, AgentStatus
from models.hardware_model import GPUSpec, GPUVendor, GPUBackend
from models.operator_ir import ClusterConfig
from knowledge_base.hardware_specs.gpu_database import GPU_DATABASE, get_gpu_spec

logger = logging.getLogger(__name__)


class HardwareProfilerAgent(BaseAgent):
    """
    硬件分析Agent

    职责：
    1. 识别集群中所有GPU类型
    2. 获取每种GPU的详细规格
    3. 分析集群的通信拓扑
    4. 为算子生成提供硬件约束信息
    5. 使用LLM解析用户的自然语言硬件描述
    """

    def __init__(self, llm_client=None, config: dict = None):
        super().__init__("HardwareProfilerAgent", llm_client, config)

    def get_system_prompt(self) -> str:
        gpu_list = "\n".join(f"  - {k}: {v.model_name} ({v.vendor.value})" for k, v in GPU_DATABASE.items())
        return f"""你是一个GPU硬件专家，专精于各厂商GPU的硬件规格分析。

当前已知的GPU型号：
{gpu_list}

你的任务是：
1. 从用户描述中识别GPU型号
2. 分析集群的硬件组成
3. 识别异构集群中不同GPU之间的性能差异
4. 建议最优的算子后端选择

请用JSON格式返回分析结果。"""

    async def run(self, context: AgentContext, **kwargs) -> AgentResult:
        self._start_timer()
        self.set_status(AgentStatus.RUNNING)

        cluster_config: Optional[ClusterConfig] = kwargs.get("cluster_config")
        hardware_description: str = kwargs.get("hardware_description", "")

        try:
            hardware_profiles: dict[str, GPUSpec] = {}

            if cluster_config:
                hardware_profiles = await self._profile_cluster(cluster_config)
            elif hardware_description:
                hardware_profiles = await self._parse_hardware_description(hardware_description)
            else:
                # 使用上下文中的GPU列表
                for gpu_id in context.target_gpus:
                    spec = get_gpu_spec(gpu_id)
                    if spec:
                        hardware_profiles[gpu_id] = spec
                    else:
                        logger.warning(f"[HardwareProfiler] Unknown GPU: {gpu_id}")

            if not hardware_profiles:
                return self.failure_result("No GPU hardware profiles could be determined")

            # 分析集群特性
            cluster_analysis = self._analyze_cluster(hardware_profiles)
            context.add_artifact("hardware_profiles", hardware_profiles)
            context.add_artifact("cluster_analysis", cluster_analysis)

            logger.info(f"[HardwareProfiler] Profiled {len(hardware_profiles)} GPU type(s): "
                       f"{list(hardware_profiles.keys())}")
            logger.info(f"[HardwareProfiler] Heterogeneous: {cluster_analysis['is_heterogeneous']}")

            return self.success_result(
                output=hardware_profiles,
                metrics={
                    "gpu_types": len(hardware_profiles),
                    "is_heterogeneous": cluster_analysis["is_heterogeneous"],
                    "vendors": cluster_analysis["vendors"],
                }
            )

        except Exception as e:
            self.set_status(AgentStatus.FAILED)
            return self.failure_result(str(e))

    async def _profile_cluster(self, cluster_config: ClusterConfig) -> dict[str, GPUSpec]:
        """从ClusterConfig中提取GPU规格"""
        profiles = {}
        for gpu_model, node_list in cluster_config.gpu_groups.items():
            spec = get_gpu_spec(gpu_model)
            if spec:
                profiles[gpu_model] = spec
            else:
                # 尝试用LLM识别
                if self.llm_client:
                    spec = await self._identify_gpu_with_llm(gpu_model)
                    if spec:
                        profiles[gpu_model] = spec
                    else:
                        logger.warning(f"[HardwareProfiler] Cannot profile GPU: {gpu_model}")
        return profiles

    async def _parse_hardware_description(self, description: str) -> dict[str, GPUSpec]:
        """用LLM解析自然语言硬件描述"""
        if self.llm_client is None:
            # 退化为关键词匹配
            return self._keyword_match_gpus(description)

        prompt = f"""分析以下GPU集群描述，提取所有GPU型号：

集群描述：
{description}

请以JSON格式返回，格式为：
{{
  "gpus": [
    {{"model_id": "h100_sxm5", "count": 8, "confidence": 0.95}},
    ...
  ]
}}

只使用已知的GPU型号ID：{list(GPU_DATABASE.keys())}"""

        try:
            response = await self.call_llm(prompt)
            import json
            data = json.loads(response)
            profiles = {}
            for gpu_info in data.get("gpus", []):
                model_id = gpu_info.get("model_id")
                confidence = gpu_info.get("confidence", 0)
                if confidence >= 0.7 and model_id:
                    spec = get_gpu_spec(model_id)
                    if spec:
                        profiles[model_id] = spec
            return profiles
        except Exception as e:
            logger.warning(f"[HardwareProfiler] LLM parsing failed: {e}, falling back to keyword match")
            return self._keyword_match_gpus(description)

    async def _identify_gpu_with_llm(self, gpu_model_str: str) -> Optional[GPUSpec]:
        """用LLM识别不在数据库中的GPU型号"""
        if self.llm_client is None:
            return None

        prompt = f"""请将以下GPU型号映射到最接近的已知GPU：

输入GPU型号: {gpu_model_str}
已知GPU列表: {list(GPU_DATABASE.keys())}

请以JSON格式返回：{{"matched_id": "a100_80gb", "confidence": 0.9}}
如果无法匹配，返回：{{"matched_id": null, "confidence": 0}}"""

        try:
            import json
            response = await self.call_llm(prompt)
            data = json.loads(response)
            matched_id = data.get("matched_id")
            confidence = data.get("confidence", 0)
            if matched_id and confidence >= 0.7:
                return get_gpu_spec(matched_id)
        except Exception:
            pass
        return None

    def _keyword_match_gpus(self, description: str) -> dict[str, GPUSpec]:
        """基于关键词的简单GPU识别"""
        desc_lower = description.lower()
        profiles = {}
        keyword_map = {
            "h100": "h100_sxm5",
            "a100": "a100_80gb",
            "mi300x": "mi300x",
            "mi300": "mi300x",
            "mi250": "mi250x",
            "gaudi3": "gaudi3",
            "gaudi 3": "gaudi3",
        }
        for keyword, gpu_id in keyword_map.items():
            if keyword in desc_lower:
                spec = get_gpu_spec(gpu_id)
                if spec:
                    profiles[gpu_id] = spec
        return profiles

    def _analyze_cluster(self, profiles: dict[str, GPUSpec]) -> dict:
        """分析集群特性"""
        if not profiles:
            return {"is_heterogeneous": False, "vendors": [], "backends": []}

        vendors = list({spec.vendor.value for spec in profiles.values()})
        all_backends: set[GPUBackend] = set()
        for spec in profiles.values():
            all_backends.update(spec.supported_backends)

        # 计算性能差异
        fp16_values = [spec.compute.fp16_tflops for spec in profiles.values()]
        perf_ratio = max(fp16_values) / min(fp16_values) if len(fp16_values) > 1 else 1.0

        # 检查是否所有GPU都支持Triton
        all_support_triton = all(spec.supports_triton() for spec in profiles.values())

        return {
            "is_heterogeneous": len(vendors) > 1 or len(profiles) > 1,
            "vendors": vendors,
            "backends": [b.value for b in all_backends],
            "all_support_triton": all_support_triton,
            "performance_ratio": perf_ratio,
            "recommended_abstraction": "triton" if all_support_triton else "per_backend",
            "memory_bandwidth_range": {
                "min": min(s.memory.bandwidth_gbps for s in profiles.values()),
                "max": max(s.memory.bandwidth_gbps for s in profiles.values()),
            },
        }
