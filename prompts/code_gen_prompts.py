"""
代码生成Agent的提示词模板
针对不同后端和不同类型的算子使用不同的提示策略
"""
import logging
from models.operator_ir import OperatorIR
from models.hardware_model import GPUSpec

logger = logging.getLogger(__name__)


def _get_fewshot_example(op_name: str, gpu_model: str, backend: str = "cuda") -> str:
    """从算子仓库中查找历史成功代码作为 few-shot 示例"""
    try:
        from operators.registry import get_registry
        reg = get_registry()

        # 优先精确匹配（同算子 + 同 GPU）
        entry = reg.lookup(op_name, gpu_model)

        # 退而求其次：同算子在其他 GPU 上的成功实现
        if entry is None or not entry.correctness_passed:
            entry = reg.find_similar(op_name, gpu_model)

        if entry and entry.correctness_passed and entry.source_code:
            # 截断到合理长度避免 prompt 过长
            code = entry.source_code[:3000]
            if len(entry.source_code) > 3000:
                code += "\n// ... (truncated)"
            return f"""
以下是在 {entry.gpu_model} 上验证通过的 {op_name} 参考实现（请参考其结构和风格）：
```{backend}
{code}
```
注意：上面的代码仅供参考风格，你需要根据目标 GPU 规格重新生成优化版本。"""
    except Exception as e:
        logger.debug(f"[Prompts] Few-shot lookup failed: {e}")

    return ""


def _get_cuda_forbidden() -> str:
    """从编译错误知识库动态生成禁用列表，降级到硬编码"""
    try:
        from knowledge_base.compile_error_kb import get_compile_error_kb
        kb = get_compile_error_kb()
        fragment = kb.generate_prompt_fragment("cuda")
        if fragment and len(fragment) > 50:
            return fragment + _CUDA_HALF2_EXAMPLES
    except Exception as e:
        logger.debug(f"[Prompts] KB unavailable, using fallback: {e}")
    return _CUDA_FORBIDDEN_FALLBACK + _CUDA_HALF2_EXAMPLES


_CUDA_HALF2_EXAMPLES = """

half2 正确写法示例：
- 加载：half2 val = *reinterpret_cast<const half2*>(&src[idx]);
- 存储：*reinterpret_cast<half2*>(&dst[idx]) = val;
- 构造：half2 val = __floats2half2_rn(a, b);
- grid计算：dim3 grid((N + block.x - 1) / block.x);
"""

_CUDA_FORBIDDEN_FALLBACK = """
严格禁止使用以下不存在的API或写法（会导致编译失败）：
- __float22half2_rn(float, float)：签名错误！正确是 __floats2half2_rn(float a, float b)
- __int22half2_rn / __float2float2_rn / __fmul2_rn / __fadd2_rn / __tanh2：均不存在
- h2tanh：不存在，用 __float2half(__tanhf(__half2float(x)))
- half4 / half8：CUDA 没有这些类型，用两个 half2 或 float4 代替
- ceildiv：不存在，用内联表达式 (N + M - 1) / M
- 不要对 reinterpret_cast 结果直接赋值，必须加 *：*reinterpret_cast<half2*>(p) = v;
- 不要在 kernel 代码里使用 fprintf/printf/stderr
- build_flags 只能包含合法的 nvcc 选项（如 -O3, -arch=sm_XX）
- 不要使用未定义的常量名如 sqrt_2_over_pi、M_SQRT2 等，直接写数值：0.7978845608f、1.4142135f
- 不要重新定义 h2exp / h2sin / h2cos / h2sqrt 等 CUDA 内置 half2 函数（已在 cuda_fp16.h 中定义）
- 不要使用 "using namespace nvcuda;"（nvcuda 不是 C++ namespace），正确写法是 "using namespace nvcuda::wmma;"
"""

_CUDA_WMMA_RULES = """
wmma 正确用法（必须严格遵守）：
- 必须 #include <mma.h> 并 using namespace nvcuda::wmma;  （注意：是 nvcuda::wmma，不是 nvcuda）
- RTX 4090 (sm_89) 只支持 16x16x16 tile，不支持 8x8x8
- accumulator fragment 是 float 类型：fragment<accumulator, 16, 16, 16, float>
- store_matrix_sync 目标指针必须是 float*，不能是 half*
- 如果不确定 wmma 用法，改用普通 shared memory tiling 实现，不要用 wmma
"""

_CUDA_JSON_FORMAT = """返回格式（JSON）：
{{
  "kernel_code": "完整的CUDA C++代码，包含kernel函数和 extern C launcher",
  "header_code": "",
  "build_flags": ["-O3", "-arch=sm_{sm}"],
  "launch_config": {{"grid_size_formula": "...", "block_size": 256, "shared_mem_bytes": 0}},
  "optimizations": ["列出应用的优化技术"],
  "estimated_efficiency": 0.85
}}

【重要】kernel_code 必须在最后包含一个 extern "C" launcher 函数，签名如下：
```cpp
extern "C" void launch_kernel(const void* input, void* output, int N) {{
    const half* in_ptr = reinterpret_cast<const half*>(input);
    half* out_ptr = reinterpret_cast<half*>(output);
    int block = 256;
    int grid = (N + block - 1) / block;
    your_kernel<<<grid, block>>>(in_ptr, out_ptr, N);
}}
```
其中 N 是元素总数。这个 launcher 会被 ctypes 调用来做数值验证。"""


def build_cuda_codegen_prompt(op_ir: OperatorIR, gpu_spec: GPUSpec) -> str:
    sm = gpu_spec.compute_capability.replace('.', '')
    op_name = op_ir.name.lower()

    # 复杂算子用更保守的策略：不强制 wmma，优先正确性
    if op_name in ("flash_attention", "matmul"):
        strategy = f"""
请生成正确可编译的 CUDA 内核，优先保证正确性，其次才是性能：
1. 【强制】不要使用 wmma / Tensor Core / nvcuda::wmma / mma_sync / load_matrix_sync 等任何 WMMA API
2. 使用普通共享内存 tiling 实现矩阵乘法（TILE_SIZE=16 或 32）
3. 使用 float 精度计算，输入输出可以是 half，但内部计算用 float
4. 确保内存访问合并
5. 处理边界情况
6. 代码必须能直接用 nvcc 编译，不依赖任何外部库
7. 不要调用任何未在代码中定义的辅助函数
8. 不要使用 __half2float4 / __float2half4 等不存在的向量转换函数
9. 所有辅助函数必须在调用前完整定义，或直接内联实现"""
    else:
        strategy = """
请生成高性能的CUDA内核，要求：
1. 使用共享内存减少全局内存访问
2. 确保内存访问合并(coalesced memory access)
3. 对于FP16运算，使用half2向量化提高吞吐量
4. 处理边界情况
5. 添加完整的启动配置注释"""

    return f"""你是一位CUDA内核编程专家，擅长为NVIDIA GPU编写高性能CUDA内核。

目标GPU规格：
- 型号: {gpu_spec.model_name}
- Compute Capability: {gpu_spec.compute_capability}
- 显存带宽: {gpu_spec.memory.bandwidth_gbps} GB/s
- FP16算力: {gpu_spec.compute.fp16_tflops} TFLOPs

算子规格：
{op_ir.to_dict()}

数学定义：
{op_ir.math_description}

参考实现（PyTorch）：
```python
{op_ir.reference_impl}
```
{strategy}
{_get_fewshot_example(op_name, gpu_spec.model_name, "cuda")}
{_get_cuda_forbidden()}
{_CUDA_JSON_FORMAT.format(sm=sm)}"""


# ── 降级版本：优先正确性，不用高级特性 ──────────────────────

CUDA_PROMPT_VERSION = "v4.0"

def build_cuda_simple_prompt(op_ir: OperatorIR, gpu_spec: GPUSpec) -> str:
    """生成简单版 CUDA prompt，禁用所有复杂优化，优先编译通过"""
    sm = gpu_spec.compute_capability.replace('.', '')
    return f"""你是一位CUDA编程专家。请用最简单、最安全的方式实现以下算子。

【最高优先级：代码必须能直接用 nvcc 编译通过】

目标GPU: {gpu_spec.model_name} (sm_{sm})

算子: {op_ir.name}
数学定义: {op_ir.math_description}

强制要求：
1. 【禁止】不要使用 half2 向量化、shared memory、wmma、Tensor Core
2. 【禁止】不要使用任何 __h2xxx / h2xxx 系列函数
3. 只使用 float 精度计算（输入输出可以是 half，内部全部转 float）
4. 每个线程处理一个元素，用最朴素的并行方式
5. 必须 #include <cuda_fp16.h>
6. 用 __half2float() 和 __float2half() 做类型转换
7. 不要定义任何辅助函数，全部内联
8. 不要使用 make_float2/make_float4 等向量类型
9. 不要生成 torch::Tensor 包装函数

{_get_cuda_forbidden()}
{_CUDA_JSON_FORMAT.format(sm=sm)}"""


def build_hip_codegen_prompt(op_ir: OperatorIR, gpu_spec: GPUSpec) -> str:
    return f"""你是一位HIP/ROCm内核编程专家，擅长为AMD GPU编写高性能HIP内核。

目标GPU规格：
- 型号: {gpu_spec.model_name}
- 架构: {gpu_spec.architecture}
- 计算单元(CU)数量: {gpu_spec.compute_units}
- 每CU核心数: {gpu_spec.cores_per_unit}
- 显存带宽: {gpu_spec.memory.bandwidth_gbps} GB/s
- FP16算力: {gpu_spec.compute.fp16_tflops} TFLOPs
- 是否有Matrix Core: {gpu_spec.compute.has_matrix_core}
- 最低ROCm版本: {gpu_spec.rocm_version_min}

算子规格：
{op_ir.to_dict()}

数学定义：
{op_ir.math_description}

参考实现（PyTorch）：
```python
{op_ir.reference_impl}
```

请生成高性能的HIP内核，要求：
1. HIP代码与CUDA高度相似，使用__hip_bfloat16等HIP类型
2. 利用Matrix Core (MFMA指令)进行矩阵计算（如果GPU支持）
3. 使用LDS（Local Data Share，相当于CUDA shared memory）
4. 保证波前(wavefront，64线程)级别的访问合并
5. 考虑CDNA架构的缓存层次（L1/L2）
6. 添加完整的hipLaunchKernelGGL调用示例

返回格式（JSON）：
{{
  "kernel_code": "完整的HIP C++代码",
  "header_code": "头文件定义（如有）",
  "build_flags": ["--offload-arch=gfx942", "-O3"],
  "launch_config": {{
    "grid_size_formula": "...",
    "block_size": 256,
    "shared_mem_bytes": 0
  }},
  "optimizations": [...],
  "estimated_efficiency": 0.80
}}"""


def build_triton_codegen_prompt(op_ir: OperatorIR, gpu_specs: list[GPUSpec]) -> str:
    hw_summary = "\n".join(
        f"  - {s.model_name}: {s.vendor.value}, BW={s.memory.bandwidth_gbps}GB/s, FP16={s.compute.fp16_tflops}TFLOPs"
        for s in gpu_specs
    )
    return f"""你是OpenAI Triton编程专家，擅长编写跨硬件的高性能Triton内核。

目标GPU集群（需要同时支持）：
{hw_summary}

算子规格：
{op_ir.to_dict()}

数学定义：
{op_ir.math_description}

参考实现（PyTorch）：
```python
{op_ir.reference_impl}
```

请生成高性能的Triton内核，要求：
1. 使用tl.load/tl.store进行分块内存访问，提高内存效率
2. 合理设置BLOCK_SIZE等配置参数，支持不同硬件调优
3. 使用@triton.autotune进行自动调优配置
4. 使用tl.dot进行矩阵乘法（会自动映射到Tensor Core/Matrix Core）
5. 实现online softmax等算法技巧（如适用）
6. 添加Python调用封装函数

返回格式（JSON）：
{{
  "kernel_code": "完整的Triton Python代码（含import）",
  "autotune_configs": [
    {{"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 3}}
  ],
  "python_wrapper": "Python调用封装函数",
  "optimizations": [...],
  "estimated_efficiency": 0.82
}}"""


def build_ascendc_codegen_prompt(op_ir: OperatorIR, gpu_spec: GPUSpec) -> str:
    return f"""你是一位华为昇腾 AscendC 算子编程专家，熟悉 DaVinci 架构和 AscendC 编程模型。

目标 GPU 规格：
- 型号: {gpu_spec.model_name}
- 架构: {gpu_spec.architecture}
- AI Core 数量: {gpu_spec.compute_units}
- HBM 带宽: {gpu_spec.memory.bandwidth_gbps} GB/s
- FP16 算力: {gpu_spec.compute.fp16_tflops} TFLOPs
- BF16 算力: {gpu_spec.compute.bf16_tflops} TFLOPs

算子规格：
{op_ir.to_dict()}

数学定义：
{op_ir.math_description}

参考实现（PyTorch）：
```python
{op_ir.reference_impl}
```

请生成高性能的 AscendC 算子内核，要求：
1. 使用 AscendC 编程框架（基于 C++ 模板，头文件 kernel_operator.h）
2. 合理使用片上内存层次：Global Memory -> L1 Buffer -> L0A/L0B/L0C（矩阵）或 UB（Vector）
3. 充分利用 Cube 单元（矩阵乘法）和 Vector 单元（逐元素操作）
4. 使用双缓冲(double buffer)流水线隐藏数据搬移延迟
5. Tile 大小必须满足 16 字节对齐（Cube 操作需要 16 的倍数）
6. 使用 DataCopy 完成 GM->L1->L0/UB 的数据搬移
7. 使用 Fixpipe/PipeBarrier 进行流水线同步
8. 使用 AscendC 内置 API：LocalTensor, GlobalTensor, TBuf, TPipe 等

AscendC 关键 API 参考：
- `TPipe pipe;` — 流水线管理
- `TBuf<QuePosition::VECIN> inQueue;` — UB 队列分配
- `DataCopy(dst, src, count);` — 数据搬移
- `Add/Mul/Exp/Sqrt(dst, src0, src1, count);` — Vector 计算
- `Mmad(dst, src0, src1, m, k, n);` — 矩阵乘法

返回格式（JSON）：
{{
  "kernel_code": "完整的 AscendC C++ 算子代码，包含 Init/Process/ComputeEach 等方法",
  "header_code": "#include <kernel_operator.h> 等头文件",
  "build_flags": ["--target=ascend910b", "-O2", "-std=c++17"],
  "launch_config": {{
    "aiv_num": {gpu_spec.compute_units},
    "tile_length": 21760,
    "double_buffer": true
  }},
  "optimizations": ["列出应用的优化技术，如双缓冲、向量化、Tile分块等"],
  "estimated_efficiency": 0.75
}}"""


def build_optimization_prompt(kernel_code: str, gpu_spec: GPUSpec, profiling_result: dict) -> str:
    return f"""你是GPU性能优化专家。请分析以下内核的性能瓶颈并提出优化建议。

目标GPU: {gpu_spec.model_name} ({gpu_spec.architecture})
- 显存带宽: {gpu_spec.memory.bandwidth_gbps} GB/s
- FP16算力: {gpu_spec.compute.fp16_tflops} TFLOPs

性能分析结果:
- 实际带宽利用率: {profiling_result.get('bandwidth_utilization', 0):.1%}
- 实际计算利用率: {profiling_result.get('compute_utilization', 0):.1%}
- 延迟: {profiling_result.get('latency_ms', 0):.3f} ms
- 主要瓶颈: {profiling_result.get('bottleneck', 'unknown')}
- L1缓存命中率: {profiling_result.get('l1_hit_rate', 0):.1%}
- 寄存器使用: {profiling_result.get('registers_per_thread', 0)}

当前内核代码:
```
{kernel_code[:3000]}
```

请提出具体的优化方案并重写内核代码。优化方向：
1. 如果内存带宽受限：增加算术强度、使用向量化访问、改进缓存利用
2. 如果计算受限：减少指令开销、使用快速数学函数、展开循环
3. 如果延迟受限：增加occupancy、隐藏内存延迟、pipeline优化

返回优化后的完整代码和优化说明（JSON格式）。"""
