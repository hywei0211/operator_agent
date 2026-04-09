"""
GPU Discovery Agent - GPU 规格自动发现
通过 MCP Server 瀑布式查询，将未知 GPU 型号转化为标准 GPUSpec
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

from agents.base_agent import BaseAgent, AgentContext, AgentResult, AgentStatus
from mcp_servers.base_server import MCPClient
from models.hardware_model import (
    GPUSpec, GPUVendor, GPUBackend, MemorySpec, ComputeSpec, InterconnectSpec
)

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredGPUInfo:
    """GPU 发现结果"""
    model_name: str
    raw_specs: dict                         # 从各数据源获取的原始数据
    normalized_spec: Optional[GPUSpec]      # 标准化后的 GPUSpec
    source: str                             # 数据来源
    confidence: float                       # 可信度 0~1
    missing_fields: list = field(default_factory=list)  # 缺失但重要的字段
    warnings: list = field(default_factory=list)


class GPUDiscoveryAgent(BaseAgent):
    """
    GPU 规格自动发现 Agent

    工作流：
    1. 标准化 GPU 名称
    2. 瀑布式查询（本地DB → MCP网络爬虫 → LLM推断）
    3. 合并多源数据，解决冲突
    4. 转换为标准 GPUSpec
    5. 标注可信度和缺失字段
    """

    def __init__(self, mcp_client: MCPClient, llm_client=None, config: dict = None):
        super().__init__("GPUDiscoveryAgent", llm_client, config)
        self.mcp = mcp_client

    def get_system_prompt(self) -> str:
        return "你是GPU硬件专家，能够识别和分析各种GPU的规格参数。"

    async def run(self, context: AgentContext, **kwargs) -> AgentResult:
        self._start_timer()
        self.set_status(AgentStatus.RUNNING)

        gpu_names: list[str] = kwargs.get("gpu_names", context.target_gpus)
        if not gpu_names:
            return self.failure_result("No GPU names provided")

        try:
            discovered: dict[str, DiscoveredGPUInfo] = {}
            for name in gpu_names:
                info = await self._discover_single(name)
                discovered[name] = info
                logger.info(
                    f"[GPUDiscovery] {name}: source={info.source}, "
                    f"confidence={info.confidence:.0%}, "
                    f"spec={'OK' if info.normalized_spec else 'PARTIAL'}"
                )

            context.add_artifact("discovered_gpus", discovered)
            hardware_profiles = {
                name: info.normalized_spec
                for name, info in discovered.items()
                if info.normalized_spec
            }
            context.add_artifact("hardware_profiles", hardware_profiles)

            return self.success_result(
                output=discovered,
                metrics={
                    "total": len(discovered),
                    "fully_resolved": sum(1 for i in discovered.values() if i.normalized_spec),
                    "avg_confidence": sum(i.confidence for i in discovered.values()) / len(discovered),
                }
            )
        except Exception as e:
            self.set_status(AgentStatus.FAILED)
            return self.failure_result(str(e))

    async def _discover_single(self, gpu_name: str) -> DiscoveredGPUInfo:
        """对单个 GPU 型号执行发现流程"""
        # 步骤1：通过 MCP 查询规格
        resp = await self.mcp.call(
            "gpu_spec_server", "search_gpu_spec",
            model_name=gpu_name
        )

        if resp.success and resp.data:
            raw = resp.data
            spec = self._dict_to_gpu_spec(raw, gpu_name)
            return DiscoveredGPUInfo(
                model_name=gpu_name,
                raw_specs=raw,
                normalized_spec=spec,
                source=raw.get("_source", resp.source),
                confidence=raw.get("_confidence", 0.8),
                missing_fields=self._check_missing_fields(raw),
            )

        # 完全未知的 GPU，返回部分信息
        return DiscoveredGPUInfo(
            model_name=gpu_name,
            raw_specs={},
            normalized_spec=None,
            source="unknown",
            confidence=0.0,
            missing_fields=["all"],
            warnings=[f"Could not find specs for {gpu_name}"],
        )

    def _dict_to_gpu_spec(self, raw: dict, gpu_name: str) -> Optional[GPUSpec]:
        """将原始 dict 转换为标准 GPUSpec"""
        try:
            vendor_map = {
                "nvidia": GPUVendor.NVIDIA,
                "amd": GPUVendor.AMD,
                "intel": GPUVendor.INTEL,
                "huawei": GPUVendor.HUAWEI,
                "apple": GPUVendor.APPLE,
            }
            vendor = vendor_map.get(raw.get("vendor", "").lower(), GPUVendor.NVIDIA)

            backend_map = {
                "cuda": GPUBackend.CUDA, "hip": GPUBackend.HIP,
                "sycl": GPUBackend.SYCL, "triton": GPUBackend.TRITON,
                "ascendc": GPUBackend.ASCENDC, "opencl": GPUBackend.OPENCL,
            }
            backends = [
                backend_map[b] for b in raw.get("supported_backends", [])
                if b in backend_map
            ]
            if not backends:
                # 根据厂商推断默认后端
                defaults = {
                    GPUVendor.NVIDIA: [GPUBackend.CUDA, GPUBackend.TRITON],
                    GPUVendor.AMD: [GPUBackend.HIP, GPUBackend.TRITON],
                    GPUVendor.INTEL: [GPUBackend.SYCL],
                    GPUVendor.HUAWEI: [GPUBackend.ASCENDC],
                }
                backends = defaults.get(vendor, [GPUBackend.OPENCL])

            return GPUSpec(
                model_name=raw.get("model_name", gpu_name),
                vendor=vendor,
                architecture=raw.get("architecture", "Unknown"),
                release_year=raw.get("release_year", 2024),
                compute_units=int(raw.get("compute_units", 0)),
                cores_per_unit=int(raw.get("cores_per_unit", 128)),
                total_cores=int(raw.get("total_cores", 0) or
                                raw.get("compute_units", 0) * raw.get("cores_per_unit", 128)),
                base_clock_mhz=int(raw.get("base_clock", 1000)),
                boost_clock_mhz=int(raw.get("boost_clock", 1500)),
                memory=MemorySpec(
                    capacity_gb=float(raw.get("memory_gb", 0)),
                    bandwidth_gbps=float(raw.get("memory_bandwidth_gbps",
                                                  raw.get("memory_bandwidth", 0))),
                    memory_type=raw.get("memory_type", "HBM"),
                    ecc_support=True,
                ),
                compute=ComputeSpec(
                    fp32_tflops=float(raw.get("fp32_tflops", 0)),
                    fp16_tflops=float(raw.get("fp16_tflops", 0)),
                    bf16_tflops=float(raw.get("bf16_tflops", 0)),
                    int8_tops=float(raw.get("int8_tops", 0)),
                    has_tensor_core=raw.get("has_tensor_core", vendor == GPUVendor.NVIDIA),
                    has_matrix_core=raw.get("has_matrix_core", vendor == GPUVendor.AMD),
                ),
                interconnect=InterconnectSpec(pcie_gen=4, pcie_lanes=16),
                supported_backends=backends,
                compute_capability=raw.get("compute_capability"),
                supports_nccl=vendor == GPUVendor.NVIDIA,
                supports_rccl=vendor == GPUVendor.AMD,
            )
        except Exception as e:
            logger.warning(f"[GPUDiscovery] Failed to convert raw spec: {e}")
            return None

    def _check_missing_fields(self, raw: dict) -> list[str]:
        """检查缺失的关键字段"""
        critical = ["fp16_tflops", "memory_bandwidth_gbps", "compute_units"]
        return [f for f in critical if not raw.get(f)]
