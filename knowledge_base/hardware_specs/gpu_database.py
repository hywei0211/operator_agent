"""
GPU硬件规格数据库
收录主流GPU的详细规格，作为代码生成和优化的依据
"""
from models.hardware_model import (
    GPUSpec, GPUVendor, GPUBackend, MemorySpec, ComputeSpec, InterconnectSpec
)


# ============================================================
# NVIDIA GPU 规格
# ============================================================

H100_SXM5 = GPUSpec(
    model_name="H100 SXM5",
    vendor=GPUVendor.NVIDIA,
    architecture="Hopper",
    release_year=2022,
    compute_units=132,          # 132个SM
    cores_per_unit=128,
    total_cores=16896,
    base_clock_mhz=1095,
    boost_clock_mhz=1830,
    memory=MemorySpec(
        capacity_gb=80,
        bandwidth_gbps=3350,
        memory_type="HBM3",
        num_memory_stacks=5,
        ecc_support=True,
    ),
    compute=ComputeSpec(
        fp32_tflops=67.0,
        fp16_tflops=989.0,
        bf16_tflops=989.0,
        int8_tops=1979.0,
        fp8_tops=3958.0,
        tf32_tflops=494.0,
        has_tensor_core=True,
    ),
    interconnect=InterconnectSpec(
        pcie_gen=5,
        pcie_lanes=16,
        nvlink_bandwidth_gbps=900,
        nvlink_version=4,
        num_links=18,
    ),
    supported_backends=[GPUBackend.CUDA, GPUBackend.TRITON],
    compute_capability="9.0",
    driver_version_min="525.85",
    supports_nccl=True,
    max_p2p_bandwidth_gbps=900,
)

H100_PCIe = GPUSpec(
    model_name="H100 PCIe",
    vendor=GPUVendor.NVIDIA,
    architecture="Hopper",
    release_year=2022,
    compute_units=114,
    cores_per_unit=128,
    total_cores=14592,
    base_clock_mhz=1095,
    boost_clock_mhz=1755,
    memory=MemorySpec(
        capacity_gb=80,
        bandwidth_gbps=2000,
        memory_type="HBM2e",
        num_memory_stacks=5,
        ecc_support=True,
    ),
    compute=ComputeSpec(
        fp32_tflops=51.0,
        fp16_tflops=756.0,
        bf16_tflops=756.0,
        int8_tops=1513.0,
        fp8_tops=3026.0,
        tf32_tflops=378.0,
        has_tensor_core=True,
    ),
    interconnect=InterconnectSpec(
        pcie_gen=5,
        pcie_lanes=16,
        nvlink_bandwidth_gbps=600,
        nvlink_version=4,
        num_links=12,
    ),
    supported_backends=[GPUBackend.CUDA, GPUBackend.TRITON],
    compute_capability="9.0",
    driver_version_min="525.85",
    supports_nccl=True,
    max_p2p_bandwidth_gbps=600,
)

A100_SXM4_80GB = GPUSpec(
    model_name="A100 SXM4 80GB",
    vendor=GPUVendor.NVIDIA,
    architecture="Ampere",
    release_year=2020,
    compute_units=108,
    cores_per_unit=128,
    total_cores=6912,
    base_clock_mhz=1095,
    boost_clock_mhz=1410,
    memory=MemorySpec(
        capacity_gb=80,
        bandwidth_gbps=2039,
        memory_type="HBM2e",
        num_memory_stacks=5,
        ecc_support=True,
    ),
    compute=ComputeSpec(
        fp32_tflops=19.5,
        fp16_tflops=312.0,
        bf16_tflops=312.0,
        int8_tops=624.0,
        tf32_tflops=156.0,
        has_tensor_core=True,
    ),
    interconnect=InterconnectSpec(
        pcie_gen=4,
        pcie_lanes=16,
        nvlink_bandwidth_gbps=600,
        nvlink_version=3,
        num_links=12,
    ),
    supported_backends=[GPUBackend.CUDA, GPUBackend.TRITON],
    compute_capability="8.0",
    driver_version_min="450.80",
    supports_nccl=True,
    max_p2p_bandwidth_gbps=600,
)

# ============================================================
# AMD GPU 规格
# ============================================================

MI300X = GPUSpec(
    model_name="Instinct MI300X",
    vendor=GPUVendor.AMD,
    architecture="CDNA3",
    release_year=2023,
    compute_units=304,          # 304个CU
    cores_per_unit=64,
    total_cores=19456,
    base_clock_mhz=900,
    boost_clock_mhz=2100,
    memory=MemorySpec(
        capacity_gb=192,        # 最大显存！
        bandwidth_gbps=5300,    # 最高带宽！
        memory_type="HBM3",
        num_memory_stacks=8,
        ecc_support=True,
    ),
    compute=ComputeSpec(
        fp32_tflops=163.4,
        fp16_tflops=1307.4,
        bf16_tflops=1307.4,
        int8_tops=2614.9,
        fp8_tops=5229.8,
        has_matrix_core=True,
    ),
    interconnect=InterconnectSpec(
        pcie_gen=5,
        pcie_lanes=16,
        infinity_fabric_gbps=896,   # 多Die间Infinity Fabric
        num_links=7,
    ),
    supported_backends=[GPUBackend.HIP, GPUBackend.TRITON, GPUBackend.OPENCL],
    rocm_version_min="6.0",
    driver_version_min="6.0",
    supports_rccl=True,
    max_p2p_bandwidth_gbps=896,
)

MI250X = GPUSpec(
    model_name="Instinct MI250X",
    vendor=GPUVendor.AMD,
    architecture="CDNA2",
    release_year=2021,
    compute_units=220,
    cores_per_unit=64,
    total_cores=14080,
    base_clock_mhz=800,
    boost_clock_mhz=1700,
    memory=MemorySpec(
        capacity_gb=128,
        bandwidth_gbps=3277,
        memory_type="HBM2e",
        num_memory_stacks=8,
        ecc_support=True,
    ),
    compute=ComputeSpec(
        fp32_tflops=47.9,
        fp16_tflops=383.0,
        bf16_tflops=383.0,
        int8_tops=383.0,
        has_matrix_core=True,
    ),
    interconnect=InterconnectSpec(
        pcie_gen=4,
        pcie_lanes=16,
        infinity_fabric_gbps=800,
        num_links=4,
    ),
    supported_backends=[GPUBackend.HIP, GPUBackend.TRITON, GPUBackend.OPENCL],
    rocm_version_min="5.3",
    driver_version_min="5.3",
    supports_rccl=True,
    max_p2p_bandwidth_gbps=800,
)

# ============================================================
# Intel GPU 规格
# ============================================================

GAUDI3 = GPUSpec(
    model_name="Gaudi 3",
    vendor=GPUVendor.INTEL,
    architecture="Gaudi3",
    release_year=2024,
    compute_units=64,           # 64个矩阵引擎
    cores_per_unit=256,
    total_cores=16384,
    base_clock_mhz=1000,
    boost_clock_mhz=1800,
    memory=MemorySpec(
        capacity_gb=96,
        bandwidth_gbps=3700,
        memory_type="HBM2e",
        num_memory_stacks=6,
        ecc_support=True,
    ),
    compute=ComputeSpec(
        fp32_tflops=125.0,
        fp16_tflops=1835.0,
        bf16_tflops=1835.0,
        int8_tops=3670.0,
        fp8_tops=7340.0,
        has_xe_matrix=True,
    ),
    interconnect=InterconnectSpec(
        pcie_gen=5,
        pcie_lanes=16,
        xelink_bandwidth_gbps=600,
        num_links=8,
    ),
    supported_backends=[GPUBackend.SYCL, GPUBackend.OPENCL],
    driver_version_min="1.3.26",
    supports_oneccl=True,
    max_p2p_bandwidth_gbps=600,
)

# ============================================================
# GPU数据库索引
# ============================================================

RTX_4090 = GPUSpec(
    model_name="GeForce RTX 4090",
    vendor=GPUVendor.NVIDIA,
    architecture="Ada Lovelace",
    release_year=2022,
    compute_units=128,          # 128个SM
    cores_per_unit=128,
    total_cores=16384,
    base_clock_mhz=2235,
    boost_clock_mhz=2520,
    memory=MemorySpec(
        capacity_gb=24,
        bandwidth_gbps=1008,
        memory_type="GDDR6X",
        ecc_support=False,
    ),
    compute=ComputeSpec(
        fp32_tflops=82.6,
        fp16_tflops=165.2,
        bf16_tflops=165.2,
        int8_tops=661.0,
        fp8_tops=1321.0,
        tf32_tflops=82.6,
        has_tensor_core=True,
    ),
    interconnect=InterconnectSpec(pcie_gen=4, pcie_lanes=16),
    supported_backends=[GPUBackend.CUDA, GPUBackend.TRITON],
    compute_capability="8.9",
    driver_version_min="520.0",
    supports_nccl=True,
    supports_rccl=False,
)


RTX_3090 = GPUSpec(
    model_name="GeForce RTX 3090",
    vendor=GPUVendor.NVIDIA,
    architecture="Ampere",
    release_year=2020,
    compute_units=82,
    cores_per_unit=128,
    total_cores=10496,
    base_clock_mhz=1395,
    boost_clock_mhz=1695,
    memory=MemorySpec(
        capacity_gb=24,
        bandwidth_gbps=936,
        memory_type="GDDR6X",
        ecc_support=False,
    ),
    compute=ComputeSpec(
        fp32_tflops=35.6,
        fp16_tflops=71.2,
        bf16_tflops=71.2,
        int8_tops=142.0,
        has_tensor_core=True,
    ),
    interconnect=InterconnectSpec(pcie_gen=4, pcie_lanes=16),
    supported_backends=[GPUBackend.CUDA, GPUBackend.TRITON],
    compute_capability="8.6",
    supports_nccl=True,
)

# ============================================================
# 华为昇腾 GPU 规格
# ============================================================

ASCEND_910B = GPUSpec(
    model_name="Ascend 910B",
    vendor=GPUVendor.HUAWEI,
    architecture="DaVinci v2",
    release_year=2023,
    compute_units=32,           # 32个AI Core
    cores_per_unit=1,
    total_cores=32,
    base_clock_mhz=1000,
    boost_clock_mhz=1000,
    memory=MemorySpec(
        capacity_gb=64,
        bandwidth_gbps=1600,
        memory_type="HBM2e",
        ecc_support=True,
    ),
    compute=ComputeSpec(
        fp32_tflops=15.0,
        fp16_tflops=256.0,
        bf16_tflops=256.0,
        int8_tops=512.0,
        has_tensor_core=False,
        has_matrix_core=True,   # Cube 矩阵单元
    ),
    interconnect=InterconnectSpec(pcie_gen=4, pcie_lanes=16),
    supported_backends=[GPUBackend.ASCENDC],
    compute_capability=None,
    supports_nccl=False,
    supports_rccl=False,
)


ASCEND_910C = GPUSpec(
    model_name="Ascend 910C",
    vendor=GPUVendor.HUAWEI,
    architecture="DaVinci v3",
    release_year=2024,
    compute_units=64,
    cores_per_unit=1,
    total_cores=64,
    base_clock_mhz=1200,
    boost_clock_mhz=1200,
    memory=MemorySpec(
        capacity_gb=96,
        bandwidth_gbps=3200,
        memory_type="HBM3",
        ecc_support=True,
    ),
    compute=ComputeSpec(
        fp32_tflops=30.0,
        fp16_tflops=800.0,
        bf16_tflops=800.0,
        int8_tops=1600.0,
        has_matrix_core=True,
    ),
    interconnect=InterconnectSpec(pcie_gen=4, pcie_lanes=16),
    supported_backends=[GPUBackend.ASCENDC],
    compute_capability=None,
    supports_nccl=False,
)


GPU_DATABASE: dict[str, GPUSpec] = {
    # NVIDIA
    "h100_sxm5": H100_SXM5,
    "h100_pcie": H100_PCIe,
    "a100_80gb": A100_SXM4_80GB,
    "rtx_4090":  RTX_4090,
    "rtx4090":   RTX_4090,      # 别名
    "rtx_3090":  RTX_3090,
    "rtx3090":   RTX_3090,
    # AMD
    "mi300x": MI300X,
    "mi250x": MI250X,
    # Intel
    "gaudi3": GAUDI3,
    # 华为昇腾
    "ascend_910b": ASCEND_910B,
    "ascend910b":  ASCEND_910B,
    "ascend_910c": ASCEND_910C,
    "ascend910c":  ASCEND_910C,
}

# 按厂商分组
NVIDIA_GPUS = {k: v for k, v in GPU_DATABASE.items() if v.vendor == GPUVendor.NVIDIA}
AMD_GPUS = {k: v for k, v in GPU_DATABASE.items() if v.vendor == GPUVendor.AMD}
INTEL_GPUS = {k: v for k, v in GPU_DATABASE.items() if v.vendor == GPUVendor.INTEL}


def get_gpu_spec(model_id: str) -> GPUSpec | None:
    """根据型号ID获取GPU规格"""
    return GPU_DATABASE.get(model_id.lower())


def find_gpus_by_vendor(vendor: GPUVendor) -> dict[str, GPUSpec]:
    return {k: v for k, v in GPU_DATABASE.items() if v.vendor == vendor}


def find_gpus_supporting_backend(backend: GPUBackend) -> dict[str, GPUSpec]:
    return {k: v for k, v in GPU_DATABASE.items() if backend in v.supported_backends}
