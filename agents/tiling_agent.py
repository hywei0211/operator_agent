"""
Tiling Agent - 自动计算最优分块配置
核心约束：每次处理的数据块必须装入片上 SRAM
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

from agents.base_agent import BaseAgent, AgentContext, AgentResult, AgentStatus
from models.hardware_model import GPUSpec, GPUVendor
from models.operator_ir import OperatorIR, OperatorCategory

logger = logging.getLogger(__name__)


@dataclass
class TilingConfig:
    """完整的 Tiling 配置（含多组候选，供 autotuner 选择）"""
    # 推荐配置
    recommended: dict = field(default_factory=dict)
    # 候选配置列表（从激进到保守）
    candidates: list = field(default_factory=list)
    # 约束说明
    constraints: dict = field(default_factory=dict)
    # 利用率预估
    estimated_sram_utilization: float = 0.0


class TilingAgent(BaseAgent):
    """
    Tiling 计算 Agent

    核心逻辑：
    1. 根据 GPU 的片上内存大小，计算每次可处理的最大数据块
    2. 满足硬件对齐约束（如昇腾 Cube 的 16×16 对齐）
    3. 生成多组候选配置（覆盖激进/均衡/保守三档）
    4. 考虑双缓冲策略（通常需要 2× SRAM 空间）
    """

    def __init__(self, llm_client=None, config: dict = None):
        super().__init__("TilingAgent", llm_client, config)

    def get_system_prompt(self) -> str:
        return "你是GPU性能优化专家，精通各种GPU的片上内存层次和分块优化策略。"

    async def run(self, context: AgentContext, **kwargs) -> AgentResult:
        self._start_timer()
        self.set_status(AgentStatus.RUNNING)

        op_ir: Optional[OperatorIR] = kwargs.get("operator_ir") or context.get_artifact("operator_ir")
        gpu_spec: Optional[GPUSpec] = kwargs.get("gpu_spec")
        sdk_context = kwargs.get("sdk_context")

        if not op_ir or not gpu_spec:
            return self.failure_result("Missing operator_ir or gpu_spec")

        try:
            tiling = self._compute_tiling(op_ir, gpu_spec, sdk_context)
            logger.info(
                f"[Tiling] {op_ir.name} on {gpu_spec.model_name}: "
                f"recommended={tiling.recommended}, "
                f"sram_util={tiling.estimated_sram_utilization:.0%}"
            )
            return self.success_result(output=tiling)
        except Exception as e:
            self.set_status(AgentStatus.FAILED)
            return self.failure_result(str(e))

    def _compute_tiling(
        self,
        op_ir: OperatorIR,
        gpu_spec: GPUSpec,
        sdk_context=None,
    ) -> TilingConfig:
        """根据硬件参数计算最优 Tiling"""
        vendor = gpu_spec.vendor

        if vendor == GPUVendor.NVIDIA:
            return self._tiling_for_nvidia(op_ir, gpu_spec)
        elif vendor == GPUVendor.AMD:
            return self._tiling_for_amd(op_ir, gpu_spec)
        elif vendor == GPUVendor.HUAWEI:
            return self._tiling_for_ascend(op_ir, gpu_spec)
        elif vendor == GPUVendor.INTEL:
            return self._tiling_for_intel(op_ir, gpu_spec)
        else:
            return self._tiling_generic(op_ir, gpu_spec)

    def _tiling_for_nvidia(self, op_ir: OperatorIR, spec: GPUSpec) -> TilingConfig:
        # NVIDIA: Shared Memory 通常 48-164KB/SM
        smem_kb = 48  # 保守估计，实际可能更大

        if op_ir.category == OperatorCategory.MATMUL or op_ir.category == OperatorCategory.ATTENTION:
            # 矩阵乘法：经典 128×128×32 分块
            candidates = [
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 3},
                {"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32, "num_warps": 4, "num_stages": 4},
                {"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
            ]
            # 验证 SRAM 约束: (BM×BK + BK×BN) × 2 bytes ≤ smem
            valid = [c for c in candidates
                     if (c["BLOCK_M"]*c["BLOCK_K"] + c["BLOCK_K"]*c["BLOCK_N"]) * 2 <= smem_kb * 1024]
            recommended = valid[0] if valid else candidates[-1]
        else:
            # 逐元素/规约：向量化，每个 block 处理大块
            block_size = min(1024, max(256, smem_kb * 1024 // (3 * 2)))
            block_size = (block_size // 128) * 128
            candidates = [
                {"BLOCK_SIZE": block_size, "num_warps": block_size // 32},
                {"BLOCK_SIZE": block_size // 2, "num_warps": block_size // 64},
            ]
            recommended = candidates[0]

        sram_used = (
            recommended.get("BLOCK_M", 0) * recommended.get("BLOCK_K", 1) +
            recommended.get("BLOCK_K", 1) * recommended.get("BLOCK_N", 0)
        ) * 2
        util = sram_used / (smem_kb * 1024) if sram_used > 0 else (
            recommended.get("BLOCK_SIZE", 256) * 3 * 2 / (smem_kb * 1024)
        )

        return TilingConfig(
            recommended=recommended,
            candidates=candidates,
            constraints={"smem_kb": smem_kb, "warp_size": 32},
            estimated_sram_utilization=min(util, 1.0),
        )

    def _tiling_for_amd(self, op_ir: OperatorIR, spec: GPUSpec) -> TilingConfig:
        # AMD: LDS 64KB/CU，wavefront = 64
        lds_kb = 64

        if op_ir.category in (OperatorCategory.MATMUL, OperatorCategory.ATTENTION):
            candidates = [
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16, "waves_per_eu": 2},
                {"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 16, "waves_per_eu": 4},
            ]
        else:
            block_size = min(1024, lds_kb * 1024 // (3 * 2))
            block_size = (block_size // 64) * 64  # wavefront 对齐
            candidates = [
                {"BLOCK_SIZE": block_size, "waves_per_eu": 2},
                {"BLOCK_SIZE": block_size // 2, "waves_per_eu": 4},
            ]
        recommended = candidates[0]

        return TilingConfig(
            recommended=recommended,
            candidates=candidates,
            constraints={"lds_kb": lds_kb, "wavefront_size": 64},
            estimated_sram_utilization=0.6,
        )

    def _tiling_for_ascend(self, op_ir: OperatorIR, spec: GPUSpec) -> TilingConfig:
        """
        昇腾 Tiling 最复杂：UB/L0A/L0B/L0C 分别约束不同操作
        """
        # 尝试获取 AI Core 规格
        from knowledge_base.hardware_specs.ascend_specs import AscendGPUSpec
        ub_kb = 256  # 默认
        l0a_kb = 64
        if isinstance(spec, AscendGPUSpec) and spec.ai_core_spec:
            ub_kb = spec.ai_core_spec.ub_size_kb
            l0a_kb = spec.ai_core_spec.l0a_size_kb

        double_buffer_factor = 2
        effective_ub = (ub_kb * 1024) // double_buffer_factor

        if op_ir.category in (OperatorCategory.MATMUL, OperatorCategory.ATTENTION):
            # 矩阵必须是 16 的倍数
            tile_k = 64
            max_mn = int((l0a_kb * 1024 / (tile_k * 2)) ** 0.5)
            max_mn = (max_mn // 16) * 16
            candidates = [
                {"tile_m": min(max_mn, 256), "tile_n": min(max_mn, 256), "tile_k": tile_k,
                 "double_buffer": True, "cube_align": 16},
                {"tile_m": min(max_mn // 2, 128), "tile_n": min(max_mn // 2, 128), "tile_k": tile_k,
                 "double_buffer": True, "cube_align": 16},
            ]
        else:
            # 逐元素：根据 UB 大小，输入+输出共 3 个 tensor
            tile_len = effective_ub // (3 * 2)  # FP16=2B
            tile_len = (tile_len // 128) * 128   # 对齐向量宽度
            candidates = [
                {"tile_length": tile_len, "double_buffer": True},
                {"tile_length": tile_len // 2, "double_buffer": True},
            ]
        recommended = candidates[0]

        return TilingConfig(
            recommended=recommended,
            candidates=candidates,
            constraints={
                "ub_kb": ub_kb, "l0a_kb": l0a_kb,
                "cube_align": 16, "vector_width": 128,
                "memory_model": "manual",
            },
            estimated_sram_utilization=0.75,
        )

    def _tiling_for_intel(self, op_ir: OperatorIR, spec: GPUSpec) -> TilingConfig:
        slm_kb = 64  # Intel SLM
        block_size = min(512, slm_kb * 1024 // (3 * 2))
        candidates = [
            {"BLOCK_SIZE": block_size, "sub_group_size": 16},
            {"BLOCK_SIZE": block_size // 2, "sub_group_size": 8},
        ]
        return TilingConfig(
            recommended=candidates[0],
            candidates=candidates,
            constraints={"slm_kb": slm_kb, "simd_width": 8},
            estimated_sram_utilization=0.6,
        )

    def _tiling_generic(self, op_ir: OperatorIR, spec: GPUSpec) -> TilingConfig:
        """未知硬件的保守通用配置"""
        candidates = [
            {"BLOCK_SIZE": 256, "note": "conservative_generic"},
            {"BLOCK_SIZE": 128, "note": "safe_generic"},
        ]
        return TilingConfig(
            recommended=candidates[0],
            candidates=candidates,
            constraints={"note": "unknown hardware, using conservative defaults"},
            estimated_sram_utilization=0.3,
        )
