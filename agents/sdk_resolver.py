"""
SDK Resolver Agent - 编程模型解析
确定目标 GPU 的最优编程 SDK，并构建代码生成所需的上下文
"""
import logging
from dataclasses import dataclass, field

from agents.base_agent import BaseAgent, AgentContext, AgentResult, AgentStatus
from mcp_servers.base_server import MCPClient
from models.hardware_model import GPUSpec, GPUVendor, GPUBackend

logger = logging.getLogger(__name__)


@dataclass
class SDKContext:
    """代码生成所需的 SDK 上下文"""
    sdk_name: str                           # cuda / hip / ascendc / sycl / triton
    language: str
    compiler: str
    kernel_decorator: str
    thread_id_expr: str
    shared_memory_syntax: str
    sync_primitive: str
    memory_management: str                  # automatic / manual
    tiling_pattern: str                     # 标准分块代码模板
    key_concepts: list = field(default_factory=list)
    extra_notes: str = ""                   # 特殊硬件注意事项


class SDKResolverAgent(BaseAgent):
    """
    SDK 解析 Agent

    职责：
    1. 根据 GPU 厂商确定编程 SDK
    2. 从 MCP Server 拉取 SDK 编程指南
    3. 构建 CodeGenAgent 所需的完整代码上下文
    4. 处理未知厂商（尝试识别 + 降级到 Triton）
    """

    def __init__(self, mcp_client: MCPClient, llm_client=None, config: dict = None):
        super().__init__("SDKResolverAgent", llm_client, config)
        self.mcp = mcp_client

    def get_system_prompt(self) -> str:
        return "你是GPU编程专家，熟悉CUDA/HIP/AscendC/SYCL/Triton等各种GPU编程框架。"

    async def run(self, context: AgentContext, **kwargs) -> AgentResult:
        self._start_timer()
        self.set_status(AgentStatus.RUNNING)

        hardware_profiles: dict[str, GPUSpec] = (
            kwargs.get("hardware_profiles") or
            context.get_artifact("hardware_profiles") or {}
        )
        if not hardware_profiles:
            return self.failure_result("No hardware profiles available")

        try:
            sdk_contexts: dict[str, SDKContext] = {}
            for gpu_id, spec in hardware_profiles.items():
                if spec is None:
                    continue
                sdk_ctx = await self._resolve_sdk(spec)
                sdk_contexts[gpu_id] = sdk_ctx
                logger.info(f"[SDKResolver] {gpu_id}: sdk={sdk_ctx.sdk_name}, "
                            f"memory_model={sdk_ctx.memory_management}")

            context.add_artifact("sdk_contexts", sdk_contexts)
            return self.success_result(
                output=sdk_contexts,
                metrics={"resolved": len(sdk_contexts)}
            )
        except Exception as e:
            self.set_status(AgentStatus.FAILED)
            return self.failure_result(str(e))

    async def _resolve_sdk(self, spec: GPUSpec) -> SDKContext:
        """为单个 GPU 确定 SDK 并获取文档"""
        # 1. 通过 MCP 获取推荐 SDK
        resp = await self.mcp.call(
            "sdk_docs_server", "get_sdk_for_vendor",
            vendor=spec.vendor.value
        )
        sdk_name = resp.data.get("sdk", "triton") if resp.success else "triton"

        # 2. 获取编程指南
        guide_resp = await self.mcp.call(
            "sdk_docs_server", "get_programming_guide",
            sdk=sdk_name
        )
        guide = guide_resp.data if guide_resp.success else {}

        # 3. 获取分块代码模板
        tiling_resp = await self.mcp.call(
            "sdk_docs_server", "get_tiling_pattern",
            sdk=sdk_name
        )
        tiling_pattern = tiling_resp.data if tiling_resp.success else ""

        # 4. 生成特定硬件的额外注意事项
        extra_notes = self._build_extra_notes(spec, sdk_name)

        return SDKContext(
            sdk_name=sdk_name,
            language=guide.get("language", "C++"),
            compiler=guide.get("compiler", "unknown"),
            kernel_decorator=guide.get("kernel_decorator", "__global__"),
            thread_id_expr=guide.get("thread_id", "blockIdx.x * blockDim.x + threadIdx.x"),
            shared_memory_syntax=guide.get("shared_memory", ""),
            sync_primitive=guide.get("sync", ""),
            memory_management=guide.get("memory_model", "automatic"),
            tiling_pattern=tiling_pattern,
            key_concepts=guide.get("key_concepts", []),
            extra_notes=extra_notes,
        )

    def _build_extra_notes(self, spec: GPUSpec, sdk_name: str) -> str:
        notes = []

        if sdk_name == "cuda" and spec.compute_capability:
            sm = spec.compute_capability.replace(".", "")
            notes.append(f"编译参数: -arch=sm_{sm}")
            if float(spec.compute_capability) >= 9.0:
                notes.append("H100: 支持 FP8，可用 __nv_fp8_e4m3 数据类型")
                notes.append("H100: 支持 TMA (Tensor Memory Accelerator) 异步内存访问")

        if sdk_name == "hip":
            notes.append("注意: wavefront = 64 threads (不是CUDA的32)")
            notes.append("MFMA指令格式: __builtin_amdgcn_mfma_f32_16x16x16f16")
            if "MI300" in spec.model_name:
                notes.append("MI300X: 192GB HBM3，超大显存适合大模型全量加载")

        if sdk_name == "ascendc":
            notes.append("关键: 必须手动 DataCopy，不能直接访问 GM 指针做计算")
            notes.append("矩阵分块固定为 16×16 的倍数（Cube 单元硬性约束）")
            notes.append("双缓冲是标配，否则 MTE 搬运会成为瓶颈")

        return "\n".join(notes)
