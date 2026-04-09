"""
GPU 硬件规格数据模型 - 描述不同厂商、不同型号的GPU能力
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class GPUVendor(str, Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    QUALCOMM = "qualcomm"
    HUAWEI = "huawei"


class GPUBackend(str, Enum):
    CUDA = "cuda"           # NVIDIA CUDA
    HIP = "hip"             # AMD ROCm/HIP
    SYCL = "sycl"           # Intel oneAPI SYCL
    METAL = "metal"         # Apple Metal
    TRITON = "triton"       # OpenAI Triton (跨平台)
    OPENCL = "opencl"       # OpenCL (通用)
    ASCENDC = "ascendc"     # 华为昇腾 AscendC
    TBE = "tbe"             # 华为昇腾 TBE (旧版)


class ComputeCapability(str, Enum):
    """NVIDIA Compute Capability 等级"""
    SM_70 = "7.0"   # Volta (V100)
    SM_75 = "7.5"   # Turing (T4)
    SM_80 = "8.0"   # Ampere (A100)
    SM_86 = "8.6"   # Ampere (A10, A30)
    SM_89 = "8.9"   # Ada Lovelace (L4, L40)
    SM_90 = "9.0"   # Hopper (H100)
    SM_100 = "10.0" # Blackwell (B100)


@dataclass
class MemorySpec:
    """显存规格"""
    capacity_gb: float
    bandwidth_gbps: float
    memory_type: str            # HBM2, HBM2e, HBM3, GDDR6X...
    num_memory_stacks: int = 0
    ecc_support: bool = True


@dataclass
class ComputeSpec:
    """计算规格"""
    fp32_tflops: float
    fp16_tflops: float
    bf16_tflops: float
    int8_tops: float
    fp8_tops: float = 0.0
    tf32_tflops: float = 0.0

    # 矩阵运算加速单元
    has_tensor_core: bool = False       # NVIDIA Tensor Core
    has_matrix_core: bool = False       # AMD Matrix Core
    has_xe_matrix: bool = False         # Intel Xe Matrix Engine
    has_amx: bool = False               # Intel AMX


@dataclass
class InterconnectSpec:
    """互联规格"""
    pcie_gen: int = 4
    pcie_lanes: int = 16
    nvlink_bandwidth_gbps: float = 0.0      # NVIDIA NVLink
    nvlink_version: int = 0
    infinity_fabric_gbps: float = 0.0       # AMD Infinity Fabric
    xelink_bandwidth_gbps: float = 0.0      # Intel Xe Link
    num_links: int = 0


@dataclass
class GPUSpec:
    """完整GPU硬件规格"""
    # 基本信息
    model_name: str
    vendor: GPUVendor
    architecture: str           # Hopper, RDNA3, Ponte Vecchio...
    release_year: int

    # 核心规格
    compute_units: int          # SM(NVIDIA) / CU(AMD) / EU(Intel)
    cores_per_unit: int
    total_cores: int
    base_clock_mhz: int
    boost_clock_mhz: int

    # 规格详情
    memory: MemorySpec
    compute: ComputeSpec
    interconnect: InterconnectSpec

    # 软件支持
    supported_backends: list[GPUBackend] = field(default_factory=list)
    compute_capability: Optional[str] = None    # NVIDIA specific
    rocm_version_min: Optional[str] = None      # AMD specific
    driver_version_min: str = ""

    # 分布式训练特性
    supports_nccl: bool = False
    supports_rccl: bool = False
    supports_oneccl: bool = False
    max_p2p_bandwidth_gbps: float = 0.0

    def get_preferred_backend(self) -> GPUBackend:
        """获取该GPU的首选编程后端"""
        if self.vendor == GPUVendor.NVIDIA:
            return GPUBackend.CUDA
        elif self.vendor == GPUVendor.AMD:
            return GPUBackend.HIP
        elif self.vendor == GPUVendor.INTEL:
            return GPUBackend.SYCL
        elif self.vendor == GPUVendor.APPLE:
            return GPUBackend.METAL
        return GPUBackend.OPENCL

    def supports_triton(self) -> bool:
        return GPUBackend.TRITON in self.supported_backends

    def peak_memory_bandwidth(self) -> float:
        return self.memory.bandwidth_gbps

    def roofline_compute_intensity(self, flops: float, bytes_accessed: float) -> float:
        """计算算术强度，用于屋顶线模型分析"""
        if bytes_accessed == 0:
            return float('inf')
        return flops / bytes_accessed

    def is_memory_bound(self, flops: float, bytes_accessed: float) -> bool:
        """判断一个操作是否受内存带宽限制"""
        arithmetic_intensity = self.roofline_compute_intensity(flops, bytes_accessed)
        peak_intensity = (self.compute.fp16_tflops * 1e12) / (self.memory.bandwidth_gbps * 1e9)
        return arithmetic_intensity < peak_intensity

    def __repr__(self) -> str:
        return (f"GPUSpec({self.model_name}, {self.vendor.value}, "
                f"FP16={self.compute.fp16_tflops}TFLOPs, "
                f"MEM={self.memory.capacity_gb}GB@{self.memory.bandwidth_gbps}GB/s)")
