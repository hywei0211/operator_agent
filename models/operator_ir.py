"""
算子中间表示 (Operator Intermediate Representation)
用于在不同硬件后端之间抽象描述算子的计算语义
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class DataType(str, Enum):
    FP64 = "fp64"
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    INT4 = "int4"
    BOOL = "bool"


class OperatorCategory(str, Enum):
    """算子分类"""
    ELEMENTWISE = "elementwise"         # 逐元素操作 (ReLU, GELU, etc.)
    REDUCTION = "reduction"             # 规约操作 (sum, max, softmax)
    MATMUL = "matmul"                   # 矩阵乘法
    ATTENTION = "attention"             # 注意力机制
    CONVOLUTION = "convolution"         # 卷积
    NORMALIZATION = "normalization"     # 归一化 (LayerNorm, RMSNorm)
    EMBEDDING = "embedding"             # 嵌入查找
    COMMUNICATION = "communication"     # 集合通信 (AllReduce, AllGather)
    MEMORY = "memory"                   # 内存操作 (copy, transpose)
    FUSED = "fused"                     # 融合算子


class ParallelStrategy(str, Enum):
    """分布式并行策略"""
    NONE = "none"                       # 单设备
    DATA_PARALLEL = "data_parallel"     # 数据并行
    TENSOR_PARALLEL = "tensor_parallel" # 张量并行
    PIPELINE_PARALLEL = "pipeline"      # 流水线并行
    SEQUENCE_PARALLEL = "sequence"      # 序列并行
    EXPERT_PARALLEL = "expert"          # MoE专家并行


@dataclass
class TensorSpec:
    """张量规格"""
    name: str
    shape: list[int | str]          # 支持符号维度，如 [batch, seq_len, hidden]
    dtype: DataType
    is_input: bool = True
    is_optional: bool = False
    memory_layout: str = "row_major"  # row_major / col_major / custom

    def num_elements_symbolic(self) -> str:
        return " * ".join(str(d) for d in self.shape)

    def bytes_per_element(self) -> int:
        dtype_bytes = {
            DataType.FP64: 8, DataType.FP32: 4, DataType.FP16: 2,
            DataType.BF16: 2, DataType.FP8_E4M3: 1, DataType.FP8_E5M2: 1,
            DataType.INT32: 4, DataType.INT16: 2, DataType.INT8: 1,
            DataType.INT4: 1, DataType.BOOL: 1,
        }
        return dtype_bytes.get(self.dtype, 4)


@dataclass
class OperatorConstraints:
    """算子约束条件"""
    # 形状约束
    shape_constraints: list[str] = field(default_factory=list)  # 如 "seq_len % 128 == 0"

    # 数值精度约束
    max_relative_error: float = 1e-5
    max_absolute_error: float = 1e-6

    # 性能约束
    min_efficiency: float = 0.5     # 相对于理论峰值的最低效率
    max_latency_ms: Optional[float] = None

    # 内存约束
    max_workspace_gb: float = 1.0


@dataclass
class OperatorIR:
    """
    算子中间表示 - 硬件无关的算子描述
    这是整个系统的核心数据结构
    """
    # 基本信息
    name: str                           # 算子名称，如 "flash_attention_v2"
    category: OperatorCategory
    description: str                    # 自然语言描述

    # 输入输出
    inputs: list[TensorSpec] = field(default_factory=list)
    outputs: list[TensorSpec] = field(default_factory=list)

    # 计算语义
    math_description: str = ""          # 数学公式描述
    reference_impl: str = ""            # PyTorch参考实现代码
    hyperparams: dict[str, Any] = field(default_factory=dict)  # 超参数

    # 反向传播
    backward_math_description: str = "" # 反向传播数学公式，如 "grad_x = grad_y * sigmoid(x) * (1 + x*(1-sigmoid(x)))"
    backward_reference_impl: str = ""   # PyTorch backward 参考实现
    saved_for_backward: list[str] = field(default_factory=list)  # forward 中需保存的张量名，如 ["x"] 或 ["Q","K","V"]

    # 复杂度分析
    flops_formula: str = ""             # FLOPs计算公式
    memory_reads_formula: str = ""      # 内存读取量公式
    memory_writes_formula: str = ""     # 内存写入量公式

    # 并行化信息
    parallel_strategy: ParallelStrategy = ParallelStrategy.NONE
    sharding_spec: dict[str, Any] = field(default_factory=dict)

    # 约束
    constraints: OperatorConstraints = field(default_factory=OperatorConstraints)

    # 元数据
    source_framework: str = "pytorch"   # pytorch / jax / tensorflow
    version: str = "1.0"
    tags: list[str] = field(default_factory=list)

    def compute_arithmetic_intensity(self, concrete_shapes: dict[str, int]) -> float:
        """根据具体形状计算算术强度"""
        try:
            flops = eval(self.flops_formula, {"__builtins__": {}}, concrete_shapes)
            reads = eval(self.memory_reads_formula, {"__builtins__": {}}, concrete_shapes)
            writes = eval(self.memory_writes_formula, {"__builtins__": {}}, concrete_shapes)
            total_bytes = (reads + writes) * 2  # 假设FP16
            return flops / total_bytes if total_bytes > 0 else float('inf')
        except Exception:
            return 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "inputs": [
                {"name": t.name, "shape": t.shape, "dtype": t.dtype.value}
                for t in self.inputs
            ],
            "outputs": [
                {"name": t.name, "shape": t.shape, "dtype": t.dtype.value}
                for t in self.outputs
            ],
            "math_description": self.math_description,
            "flops_formula": self.flops_formula,
            "parallel_strategy": self.parallel_strategy.value,
        }


@dataclass
class GeneratedKernel:
    """代码生成Agent的输出 - 针对特定硬件的生成内核"""
    operator_name: str
    backend: str                    # cuda / hip / sycl / triton
    target_gpu: str                 # 目标GPU型号
    source_code: str                # 生成的源代码
    header_code: str = ""           # 头文件代码（如有）
    build_flags: list[str] = field(default_factory=list)    # 编译标志
    launch_config: dict = field(default_factory=dict)       # 启动配置(grid, block)

    # 性能预测
    estimated_flops: float = 0.0
    estimated_bandwidth_utilization: float = 0.0

    # 优化记录
    optimizations_applied: list[str] = field(default_factory=list)
    iteration: int = 1              # 优化迭代次数

    # 验证结果
    correctness_verified: bool = False
    benchmark_results: dict = field(default_factory=dict)
    verification_level: str = "none"  # none/static/llm_review/cpu_math/compiled/hw_verified/benchmarked

    # 反向传播内核
    backward_source_code: str = ""      # 生成的 backward kernel 代码
    backward_build_flags: list[str] = field(default_factory=list)


@dataclass
class ClusterConfig:
    """异构GPU集群配置"""
    cluster_name: str
    nodes: list[dict]               # 每个节点的信息
    gpu_groups: dict[str, list[str]]  # GPU型号 -> 节点列表映射

    # 通信拓扑
    intra_node_bandwidth_gbps: float = 600.0    # NVLink / Infinity Fabric
    inter_node_bandwidth_gbps: float = 400.0    # InfiniBand / RoCE
    communication_backend: str = "nccl"

    # 并行配置
    tensor_parallel_degree: int = 1
    pipeline_parallel_degree: int = 1
    data_parallel_degree: int = 1

    def total_gpus(self) -> int:
        return sum(len(gpus) for gpus in self.gpu_groups.values())

    def is_heterogeneous(self) -> bool:
        return len(self.gpu_groups) > 1
