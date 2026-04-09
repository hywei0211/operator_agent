"""
SDK 文档 MCP Server
为代码生成 Agent 提供各厂商 GPU 编程 SDK 的文档、示例和 API 参考
"""
import logging
from mcp_servers.base_server import BaseMCPServer, MCPTool

logger = logging.getLogger(__name__)

# 内置 SDK 编程模型知识库
SDK_KNOWLEDGE = {
    "cuda": {
        "language": "C++ with CUDA extensions",
        "compiler": "nvcc",
        "kernel_decorator": "__global__",
        "thread_id": "threadIdx.x + blockIdx.x * blockDim.x",
        "shared_memory": "__shared__ float smem[SIZE];",
        "sync": "__syncthreads();",
        "memory_model": "automatic",
        "matrix_api": "wmma:: (for Tensor Core) or mma PTX",
        "key_concepts": [
            "Grid/Block/Thread 三级并行层次",
            "Global Memory 由所有 thread 共享",
            "Shared Memory 在同一 block 内共享，需要 __syncthreads()",
            "寄存器是 thread 私有",
            "内存合并访问（Coalesced Access）是性能关键",
        ],
        "tiling_pattern": """
// CUDA 分块矩阵乘法模式
__global__ void matmul(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,  by = blockIdx.y;
    float acc = 0.0f;
    for (int k = 0; k < N/TILE; k++) {
        As[ty][tx] = A[(by*TILE+ty)*N + k*TILE+tx];
        Bs[ty][tx] = B[(k*TILE+ty)*N + bx*TILE+tx];
        __syncthreads();
        for (int i = 0; i < TILE; i++) acc += As[ty][i] * Bs[i][tx];
        __syncthreads();
    }
    C[(by*TILE+ty)*N + bx*TILE+tx] = acc;
}""",
    },
    "hip": {
        "language": "C++ with HIP extensions (CUDA-compatible syntax)",
        "compiler": "hipcc",
        "kernel_decorator": "__global__",
        "thread_id": "hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x",
        "shared_memory": "__shared__ float smem[SIZE];",
        "sync": "__syncthreads();",
        "memory_model": "automatic",
        "matrix_api": "rocblas / hipblaslt / MFMA intrinsics for Matrix Core",
        "key_concepts": [
            "与 CUDA 语法高度相似，大多数 CUDA 代码可直接 hipify 转换",
            "Wavefront = 64 threads（CUDA Warp = 32 threads）",
            "LDS（Local Data Share）= CUDA Shared Memory，64KB/CU",
            "MFMA 指令（__builtin_amdgcn_mfma_*）访问 Matrix Core",
            "使用 --offload-arch=gfx942 指定 MI300X 架构",
        ],
        "tiling_pattern": """
// HIP kernel 与 CUDA 几乎相同
__global__ void kernel(half* x, half* out, int N) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < N) out[idx] = __float2half(__half2float(x[idx]) * 2.0f);
}""",
    },
    "ascendc": {
        "language": "AscendC (C++ subset for Ascend AI Core)",
        "compiler": "atc (Ascend Tensor Compiler)",
        "kernel_decorator": "__aicore__ (replaces __global__)",
        "thread_id": "GetBlockIdx()  // AI Core index, not thread index",
        "shared_memory": "TBuf<TPosition::VECCALC> ubBuf; // UB Unified Buffer",
        "sync": "// No explicit sync needed within single AI Core",
        "memory_model": "MANUAL - must DataCopy between GM/L1/L0/UB",
        "matrix_api": "Matmul() with L0A/L0B/L0C buffers",
        "key_concepts": [
            "每个 AI Core 独立执行，用 GetBlockIdx() 区分",
            "必须手动 DataCopy: GM→UB(向量计算) 或 GM→L1→L0A/B(矩阵计算)",
            "UB (Unified Buffer) 用于 Vector 单元，大小 256KB",
            "L0A/B/C 专供 Cube 矩阵单元，固定 16×16 分块对齐",
            "双缓冲 (Double Buffer) 是隐藏搬运延迟的标准手段",
            "TQue 管理 UB 上的流水线队列",
        ],
        "tiling_pattern": """
// AscendC 标准逐元素算子模式
class MyKernel {
    __aicore__ inline void Init(GM_ADDR x_gm, GM_ADDR out_gm, int N) {
        xGm.SetGlobalBuffer((__gm__ half*)x_gm, N);
        outGm.SetGlobalBuffer((__gm__ half*)out_gm, N);
        pipe.InitBuffer(inQ, 2, TILE * sizeof(half));   // 双缓冲
        pipe.InitBuffer(outQ, 2, TILE * sizeof(half));
    }
    __aicore__ inline void Process() {
        for (int i = 0; i < loops; i++) {
            // 1. GM → UB
            auto x = inQ.AllocTensor<half>();
            DataCopy(x, xGm[i*TILE], TILE);
            inQ.EnQue(x);
            // 2. 向量计算
            auto xi = inQ.DeQue<half>();
            auto yo = outQ.AllocTensor<half>();
            /* 在此用 Mul/Add/Muls 等向量指令计算 */
            outQ.EnQue(yo);
            // 3. UB → GM
            auto y = outQ.DeQue<half>();
            DataCopy(outGm[i*TILE], y, TILE);
            outQ.FreeTensor(y);  inQ.FreeTensor(xi);
        }
    }
};""",
    },
    "sycl": {
        "language": "SYCL 2020 (C++17 standard)",
        "compiler": "icpx (Intel DPC++)",
        "kernel_decorator": "q.submit([&](sycl::handler& h){ h.parallel_for(...) })",
        "thread_id": "item.get_global_id(0)",
        "shared_memory": "sycl::local_accessor<float, 1> local_mem(size, h);",
        "sync": "sycl::group_barrier(item.get_group());",
        "memory_model": "automatic (USM unified shared memory)",
        "matrix_api": "sycl::ext::intel::experimental::matrix::*",
        "key_concepts": [
            "基于 C++17 标准，跨 Intel/NVIDIA/AMD 可移植",
            "nd_range 定义全局/局部工作项",
            "local_accessor 实现工作组内共享内存",
            "USM (Unified Shared Memory) 统一内存模型",
        ],
        "tiling_pattern": """
// SYCL 标准 kernel 模式
q.submit([&](sycl::handler& h) {
    auto local_mem = sycl::local_accessor<float, 1>(TILE, h);
    h.parallel_for(
        sycl::nd_range<1>(global_size, local_size),
        [=](sycl::nd_item<1> item) {
            int gid = item.get_global_id(0);
            local_mem[item.get_local_id(0)] = x[gid];
            sycl::group_barrier(item.get_group());
            out[gid] = local_mem[item.get_local_id(0)] * 2.0f;
        });
});""",
    },
    "triton": {
        "language": "Python with Triton JIT",
        "compiler": "triton (JIT, no separate compile step)",
        "kernel_decorator": "@triton.jit",
        "thread_id": "tl.program_id(axis=0)",
        "shared_memory": "# Triton manages SRAM automatically via tiling",
        "sync": "tl.debug_barrier()  # rarely needed",
        "memory_model": "automatic (compiler manages SRAM)",
        "matrix_api": "tl.dot()  # maps to Tensor Core/Matrix Core automatically",
        "key_concepts": [
            "Python 语法，最易上手，可移植到 NVIDIA/AMD",
            "tl.load/tl.store 进行分块内存访问，自动向量化",
            "tl.dot 自动映射到硬件矩阵加速单元",
            "@triton.autotune 自动搜索最优配置",
            "BLOCK_SIZE 等参数在编译时确定 (constexpr)",
        ],
        "tiling_pattern": """
@triton.autotune(
    configs=[triton.Config({'BLOCK_SIZE': bs}, num_warps=w)
             for bs in [64, 128, 256] for w in [4, 8]],
    key=['N'],
)
@triton.jit
def kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * 2.0, mask=mask)""",
    },
}

VENDOR_SDK_MAP = {
    "nvidia": "cuda",
    "amd": "hip",
    "intel": "sycl",
    "huawei": "ascendc",
    "apple": "metal",
}


class SDKDocsMCPServer(BaseMCPServer):

    def __init__(self, llm_client=None):
        super().__init__("sdk_docs_server")
        self.llm_client = llm_client

    def setup(self):
        self.register_tool(MCPTool(
            name="get_sdk_for_vendor",
            description="根据 GPU 厂商获取推荐的编程 SDK",
            parameters={"vendor": {"type": "string"}},
            handler=lambda vendor: {
                "sdk": VENDOR_SDK_MAP.get(vendor.lower(), "triton"),
                "fallback": "triton"
            },
        ))
        self.register_tool(MCPTool(
            name="get_programming_guide",
            description="获取指定 SDK 的编程指南和代码模式",
            parameters={"sdk": {"type": "string"}},
            handler=self._get_programming_guide,
        ))
        self.register_tool(MCPTool(
            name="get_tiling_pattern",
            description="获取指定 SDK 的标准分块代码模板",
            parameters={"sdk": {"type": "string"}},
            handler=lambda sdk: SDK_KNOWLEDGE.get(sdk, {}).get("tiling_pattern", ""),
        ))
        self.register_tool(MCPTool(
            name="identify_sdk_from_description",
            description="从 GPU 描述文本中识别编程 SDK",
            parameters={"description": {"type": "string"}},
            handler=self._identify_sdk,
        ))

    def _get_programming_guide(self, sdk: str) -> dict:
        info = SDK_KNOWLEDGE.get(sdk.lower())
        if not info:
            return {"error": f"Unknown SDK: {sdk}",
                    "available": list(SDK_KNOWLEDGE.keys())}
        return {k: v for k, v in info.items() if k != "tiling_pattern"}

    def _identify_sdk(self, description: str) -> dict:
        desc_lower = description.lower()
        if any(kw in desc_lower for kw in ["nvidia", "cuda", "sm_", "tensor core"]):
            return {"sdk": "cuda", "confidence": 0.95}
        if any(kw in desc_lower for kw in ["amd", "hip", "rocm", "cu ", "wavefront"]):
            return {"sdk": "hip", "confidence": 0.95}
        if any(kw in desc_lower for kw in ["huawei", "ascend", "aicore", "cann", "davinci"]):
            return {"sdk": "ascendc", "confidence": 0.95}
        if any(kw in desc_lower for kw in ["intel", "sycl", "xe ", "gaudi"]):
            return {"sdk": "sycl", "confidence": 0.90}
        return {"sdk": "triton", "confidence": 0.5, "reason": "Fallback cross-platform"}
