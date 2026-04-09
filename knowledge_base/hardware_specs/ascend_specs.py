"""
华为昇腾 GPU 硬件规格
包含昇腾 910B / 910C / 310P 等系列的详细规格和编程模型参数
"""
from dataclasses import dataclass, field
from models.hardware_model import (
    GPUSpec, GPUVendor, GPUBackend, MemorySpec, ComputeSpec, InterconnectSpec
)


# ============================================================
# 昇腾特有的硬件参数（CUDA 规格里没有的）
# ============================================================

@dataclass
class AscendCoreSpec:
    """
    AI Core 内部结构 - 昇腾特有
    这些参数直接决定了算子的 tiling 策略
    """
    # 片上缓冲区大小（算子 tiling 的核心约束）
    l0a_size_kb: int        # 矩阵左操作数缓冲（专供 Cube 单元）
    l0b_size_kb: int        # 矩阵右操作数缓冲（专供 Cube 单元）
    l0c_size_kb: int        # 矩阵输出缓冲
    l1_size_kb: int         # 中间缓冲区（GM → L1 → L0）
    ub_size_kb: int         # 统一缓冲区（专供 Vector 单元）

    # Cube 单元（矩阵乘法）规格
    cube_fp16_ops_per_cycle: int    # 每 cycle 的 FP16 矩阵计算量
    cube_tile_m: int = 16           # 矩阵分块 M 维度（固定）
    cube_tile_n: int = 16           # 矩阵分块 N 维度（固定）
    cube_tile_k: int = 16           # 矩阵分块 K 维度（固定）

    # Vector 单元规格
    vector_fp16_width: int = 128    # 向量宽度（同时处理多少个 FP16）
    vector_fp32_width: int = 64

    # MTE（内存搬运引擎）
    mte_bandwidth_gbps: float = 0.0  # 片上内存搬运带宽

    def max_matmul_m(self) -> int:
        """根据 L0A 大小计算最大可处理的 M 维度"""
        return (self.l0a_size_kb * 1024) // (self.cube_tile_k * 2)  # FP16=2bytes

    def max_matmul_n(self) -> int:
        """根据 L0B 大小计算最大可处理的 N 维度"""
        return (self.l0b_size_kb * 1024) // (self.cube_tile_k * 2)

    def ub_can_hold_elements(self, dtype_bytes: int = 2) -> int:
        """UB 能容纳的最大元素数量"""
        return (self.ub_size_kb * 1024) // dtype_bytes


@dataclass
class AscendGPUSpec(GPUSpec):
    """
    昇腾 GPU 完整规格（扩展基础 GPUSpec）
    """
    # 昇腾特有字段
    ai_core_spec: AscendCoreSpec = None
    cann_version_min: str = "7.0"           # 最低 CANN 版本
    ascendc_supported: bool = True          # 是否支持 AscendC（新编程模型）
    tbe_supported: bool = True              # 是否支持 TBE（旧编程模型）
    dynamic_shape_support: bool = True      # 是否支持动态 shape
    hccl_version: str = "2.0"              # HCCL（昇腾通信库）版本

    def get_preferred_backend(self) -> GPUBackend:
        return GPUBackend.ASCENDC  # 新增后端类型


# ============================================================
# 昇腾 910B（主流训练卡）规格
# ============================================================

ASCEND_910B = AscendGPUSpec(
    model_name="Ascend 910B",
    vendor=GPUVendor.HUAWEI,
    architecture="DaVinci v2",
    release_year=2023,
    compute_units=24,           # 24个 AI Core
    cores_per_unit=1,           # 每个AI Core是独立完整单元
    total_cores=24,
    base_clock_mhz=1000,
    boost_clock_mhz=1800,
    memory=MemorySpec(
        capacity_gb=64,
        bandwidth_gbps=1600,     # HBM2e
        memory_type="HBM2e",
        num_memory_stacks=4,
        ecc_support=True,
    ),
    compute=ComputeSpec(
        fp32_tflops=7.9,
        fp16_tflops=320.0,
        bf16_tflops=320.0,
        int8_tops=640.0,
        has_tensor_core=False,   # 昇腾叫 Cube 单元，不是 Tensor Core
    ),
    interconnect=InterconnectSpec(
        pcie_gen=4,
        pcie_lanes=16,
        # 昇腾使用 HCCS（华为高速缓存一致性互联）
        nvlink_bandwidth_gbps=0,
        num_links=0,
    ),
    supported_backends=[GPUBackend.ASCENDC, GPUBackend.TBE],
    driver_version_min="23.0",
    cann_version_min="7.0",
    # AI Core 片上结构（tiling 优化的关键参数）
    ai_core_spec=AscendCoreSpec(
        l0a_size_kb=64,
        l0b_size_kb=64,
        l0c_size_kb=256,
        l1_size_kb=1024,        # 1MB L1 buffer
        ub_size_kb=256,         # 256KB 统一缓冲区
        cube_fp16_ops_per_cycle=4096,
        cube_tile_m=16,
        cube_tile_n=16,
        cube_tile_k=16,
        vector_fp16_width=128,
        vector_fp32_width=64,
        mte_bandwidth_gbps=512,
    ),
    max_p2p_bandwidth_gbps=56,  # HCCS 带宽
)

# ============================================================
# 昇腾 910C（最新型号，性能大幅提升）规格
# ============================================================

ASCEND_910C = AscendGPUSpec(
    model_name="Ascend 910C",
    vendor=GPUVendor.HUAWEI,
    architecture="DaVinci v3",
    release_year=2024,
    compute_units=32,
    cores_per_unit=1,
    total_cores=32,
    base_clock_mhz=1200,
    boost_clock_mhz=2000,
    memory=MemorySpec(
        capacity_gb=64,
        bandwidth_gbps=2400,
        memory_type="HBM3",
        num_memory_stacks=4,
        ecc_support=True,
    ),
    compute=ComputeSpec(
        fp32_tflops=12.0,
        fp16_tflops=512.0,
        bf16_tflops=512.0,
        int8_tops=1024.0,
        fp8_tops=2048.0,
        has_tensor_core=False,
    ),
    interconnect=InterconnectSpec(
        pcie_gen=5,
        pcie_lanes=16,
        num_links=0,
    ),
    supported_backends=[GPUBackend.ASCENDC, GPUBackend.TBE],
    driver_version_min="24.0",
    cann_version_min="8.0",
    ai_core_spec=AscendCoreSpec(
        l0a_size_kb=128,
        l0b_size_kb=128,
        l0c_size_kb=512,
        l1_size_kb=2048,
        ub_size_kb=512,
        cube_fp16_ops_per_cycle=8192,
        cube_tile_m=16,
        cube_tile_n=16,
        cube_tile_k=32,         # 910C 的 K 维增大
        vector_fp16_width=256,
        vector_fp32_width=128,
        mte_bandwidth_gbps=1024,
    ),
    max_p2p_bandwidth_gbps=100,
)
