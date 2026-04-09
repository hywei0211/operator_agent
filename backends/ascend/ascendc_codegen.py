"""
AscendC 代码生成器
将 OperatorIR 翻译为针对昇腾 AI Core 优化的 AscendC C++ 代码

关键概念映射：
  CUDA                    →  AscendC
  ────────────────────────────────────
  __global__ void kernel  →  __aicore__ void kernel (每个AI Core独立执行)
  threadIdx / blockIdx    →  GetBlockIdx() (AI Core编号)
  __shared__ memory       →  TBuf<VECCALC> (UB统一缓冲区)
  cudaMemcpy              →  DataCopy (显式DMA搬运)
  tensor core WMMA        →  Matmul + L0A/L0B/L0C buffers
  atomicAdd               →  AscendC没有，需要重新设计
"""
from dataclasses import dataclass
from models.operator_ir import OperatorIR, OperatorCategory


@dataclass
class TilingConfig:
    """
    昇腾算子的 Tiling 配置
    这是昇腾算子开发中最重要的优化步骤：
    确定每次处理的数据块大小，使其恰好放入片上缓冲区
    """
    # 基础分块大小
    tile_length: int = 512      # 每个tile处理的元素数（受UB大小约束）
    # 矩阵专用分块（受L0A/L0B大小约束）
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 64
    # AI Core 数量（并行度）
    num_cores: int = 24
    # 双缓冲（流水线隐藏延迟）
    double_buffer: bool = True


def compute_tiling(op_ir: OperatorIR, ub_size_kb: int, l0_size_kb: int, num_cores: int) -> TilingConfig:
    """
    自动计算最优 Tiling 参数

    核心约束：
    1. 每次搬入 UB 的数据量 ≤ UB 大小（通常留 2/3 用于双缓冲）
    2. 矩阵分块需满足 Cube 单元的对齐要求（16 的倍数）
    3. 双缓冲时实际可用空间减半
    """
    ub_bytes = ub_size_kb * 1024
    # 双缓冲：每块只用一半
    effective_ub = ub_bytes // 2 if True else ub_bytes
    # FP16 每个元素 2 字节，输入+输出共用 UB
    tile_length = effective_ub // (2 * 3)  # 3 个 tensor (x, w, out)
    # 对齐到 128（向量宽度）
    tile_length = (tile_length // 128) * 128

    # 矩阵分块：受 L0A/L0B 约束
    l0_bytes = l0_size_kb * 1024
    # L0A = M×K×2, L0B = K×N×2, 需同时满足
    max_mn = int((l0_bytes / 2) ** 0.5) // 16 * 16
    tile_m = min(max_mn, 256)
    tile_n = min(max_mn, 256)
    tile_k = 64  # 经验值，昇腾 K 方向通常用 64

    return TilingConfig(
        tile_length=max(tile_length, 128),
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        num_cores=num_cores,
        double_buffer=True,
    )


# ============================================================
# 各类算子的 AscendC 代码模板
# ============================================================

def gen_elementwise_kernel(op_ir: OperatorIR, tiling: TilingConfig) -> str:
    """
    生成逐元素算子的 AscendC 代码
    适用于：GELU、ReLU、SiLU、逐元素加法等
    """
    op_name = op_ir.name
    math_desc = op_ir.math_description

    return f'''/**
 * AscendC Elementwise Kernel: {op_name}
 * Math: {math_desc}
 *
 * 昇腾内存流：
 *   GM (HBM) → [MTE1 DataCopy] → UB → [Vector单元计算] → UB → [MTE3 DataCopy] → GM
 *
 * Tiling: tile_length={tiling.tile_length}, num_cores={tiling.num_cores}
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t TILE_LENGTH = {tiling.tile_length};
constexpr int32_t BUFFER_NUM  = {"2" if tiling.double_buffer else "1"};  // 双缓冲

class {op_name.title().replace("_", "")}Kernel {{
public:
    __aicore__ inline {op_name.title().replace("_", "")}Kernel() {{}}

    __aicore__ inline void Init(
        GM_ADDR x_gm, GM_ADDR out_gm,
        int32_t total_length)
    {{
        this->total_length = total_length;
        // AI Core 数据分片：每个 Core 处理一段
        int32_t per_core = (total_length + GetBlockNum() - 1) / GetBlockNum();
        this->core_length = min(per_core, total_length - GetBlockIdx() * per_core);
        this->core_offset = GetBlockIdx() * per_core;

        // 绑定全局内存（GM）
        xGm.SetGlobalBuffer((__gm__ half*)x_gm + core_offset, core_length);
        outGm.SetGlobalBuffer((__gm__ half*)out_gm + core_offset, core_length);

        // 在 UB 上申请缓冲区（双缓冲隐藏搬运延迟）
        pipe.InitBuffer(inQueueX,  BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(outQueue,  BUFFER_NUM, TILE_LENGTH * sizeof(half));
    }}

    __aicore__ inline void Process() {{
        int32_t loop_count = (core_length + TILE_LENGTH - 1) / TILE_LENGTH;
        for (int32_t i = 0; i < loop_count; i++) {{
            int32_t cur_len = min(TILE_LENGTH, core_length - i * TILE_LENGTH);
            CopyIn(i, cur_len);   // GM → UB
            Compute(i, cur_len);  // UB 向量计算
            CopyOut(i, cur_len);  // UB → GM
        }}
    }}

private:
    __aicore__ inline void CopyIn(int32_t progress, int32_t cur_len) {{
        // 从 GM 搬数据到 UB（MTE1 引擎）
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * TILE_LENGTH], cur_len);
        inQueueX.EnQue(xLocal);
    }}

    __aicore__ inline void Compute(int32_t progress, int32_t cur_len) {{
        LocalTensor<half> xLocal  = inQueueX.DeQue<half>();
        LocalTensor<half> outLocal = outQueue.AllocTensor<half>();

        // ★ 在此实现: {math_desc}
        // Vector 单元指令（同时处理 {tiling.tile_length} 个元素）
        // TODO: 根据具体算子替换为对应指令
        // 示例（GELU近似）:
        //   Muls(tmp, xLocal, 0.044715f, cur_len);
        //   Mul(tmp, tmp, xLocal, cur_len);   // 0.044715 * x^2
        //   Add(tmp, tmp, one, cur_len);       // 1 + 0.044715*x^2
        //   Mul(tmp, tmp, xLocal, cur_len);   // x * (...)
        //   Muls(tmp, tmp, sqrt(2/pi), cur_len);
        //   Tanh(tmp, tmp, cur_len);
        //   ... (完整实现由LLM生成)
        Adds(outLocal, xLocal, (half)0.0f, cur_len);  // 占位

        outQueue.EnQue<half>(outLocal);
        inQueueX.FreeTensor(xLocal);
    }}

    __aicore__ inline void CopyOut(int32_t progress, int32_t cur_len) {{
        // 从 UB 搬结果回 GM（MTE3 引擎）
        LocalTensor<half> outLocal = outQueue.DeQue<half>();
        DataCopy(outGm[progress * TILE_LENGTH], outLocal, cur_len);
        outQueue.FreeTensor(outLocal);
    }}

    // GM (片外) 全局内存张量
    GlobalTensor<half> xGm, outGm;
    // UB (片上) 缓冲队列（双缓冲流水线）
    TPipe pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    int32_t total_length, core_length, core_offset;
}};

// Kernel 入口（对应 CUDA 的 __global__）
extern "C" __global__ __aicore__ void {op_name}_kernel(
    GM_ADDR x, GM_ADDR out, int32_t total_length)
{{
    {op_name.title().replace("_", "")}Kernel op;
    op.Init(x, out, total_length);
    op.Process();
}}

// Python 调用封装（通过 PyACL 调用）
/*
import acl
def launch_{op_name}(x_tensor, out_tensor, stream):
    kernel_name = "{op_name}_kernel"
    # 编译时由 atc 工具生成 .om 文件
    # acl.rt.launch_kernel(kernel_name, num_cores, args, stream)
*/
'''


def gen_matmul_kernel(op_ir: OperatorIR, tiling: TilingConfig) -> str:
    """
    生成矩阵乘法算子的 AscendC 代码
    昇腾矩阵乘法必须经过 L0A/L0B/L0C，与 CUDA 的 WMMA 类似但更底层
    """
    op_name = op_ir.name
    return f'''/**
 * AscendC MatMul Kernel: {op_name}
 * 内存流：GM → L1 → L0A/L0B → [Cube单元] → L0C → UB → GM
 *
 * Tiling: M={tiling.tile_m}, N={tiling.tile_n}, K={tiling.tile_k}
 *
 * 关键约束:
 *   L0A 容量 = tile_m × tile_k × 2 bytes ≤ L0A_SIZE
 *   L0B 容量 = tile_k × tile_n × 2 bytes ≤ L0B_SIZE
 *   L0C 容量 = tile_m × tile_n × 4 bytes ≤ L0C_SIZE (FP32 累加)
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"    // 昇腾 Matmul API
using namespace AscendC;
using namespace matmul;

// Tiling 参数（由 tiling 工具自动生成，这里硬编码示例）
constexpr int32_t TILE_M = {tiling.tile_m};
constexpr int32_t TILE_N = {tiling.tile_n};
constexpr int32_t TILE_K = {tiling.tile_k};

class {op_name.title().replace("_", "")}Kernel {{
public:
    __aicore__ inline void Init(
        GM_ADDR a_gm, GM_ADDR b_gm, GM_ADDR c_gm,
        int32_t M, int32_t N, int32_t K)
    {{
        this->M = M; this->N = N; this->K = K;
        aGm.SetGlobalBuffer((__gm__ half*)a_gm, M * K);
        bGm.SetGlobalBuffer((__gm__ half*)b_gm, K * N);
        cGm.SetGlobalBuffer((__gm__ float*)c_gm, M * N);

        // 昇腾矩阵乘法需要 5 级缓冲区
        pipe.InitBuffer(aL1Buf,  TILE_M * TILE_K * sizeof(half));   // A → L1
        pipe.InitBuffer(bL1Buf,  TILE_K * TILE_N * sizeof(half));   // B → L1
        pipe.InitBuffer(aL0Buf,  TILE_M * TILE_K * sizeof(half));   // L1 → L0A
        pipe.InitBuffer(bL0Buf,  TILE_K * TILE_N * sizeof(half));   // L1 → L0B
        pipe.InitBuffer(cL0Buf,  TILE_M * TILE_N * sizeof(float));  // 结果 L0C
        pipe.InitBuffer(cUBBuf,  TILE_M * TILE_N * sizeof(float));  // L0C → UB → GM
    }}

    __aicore__ inline void Process() {{
        int32_t m_tiles = (M + TILE_M - 1) / TILE_M;
        int32_t n_tiles = (N + TILE_N - 1) / TILE_N;

        for (int32_t mi = 0; mi < m_tiles; mi++) {{
            for (int32_t ni = 0; ni < n_tiles; ni++) {{
                // 初始化 L0C 为 0
                LocalTensor<float> cL0 = cL0Buf.Get<float>();
                Duplicate(cL0, 0.0f, TILE_M * TILE_N);

                for (int32_t ki = 0; ki < (K + TILE_K - 1) / TILE_K; ki++) {{
                    // Step1: GM → L1（两路同时搬，利用 MTE1+MTE2）
                    LocalTensor<half> aL1 = aL1Buf.Get<half>();
                    LocalTensor<half> bL1 = bL1Buf.Get<half>();
                    DataCopy(aL1, aGm[mi*TILE_M*K + ki*TILE_K], TILE_M * TILE_K);
                    DataCopy(bL1, bGm[ki*TILE_K*N + ni*TILE_N], TILE_K * TILE_N);

                    // Step2: L1 → L0A/L0B（格式转换，满足 Cube 单元的数据排布要求）
                    LocalTensor<half> aL0 = aL0Buf.Get<half>();
                    LocalTensor<half> bL0 = bL0Buf.Get<half>();
                    LoadData(aL0, aL1, {{TILE_M, TILE_K}});  // 转为 Cube 要求的 Zz 格式
                    LoadData(bL0, bL1, {{TILE_K, TILE_N}});  // 转为 nZ 格式

                    // Step3: Cube 计算 C += A × B（核心！对应 CUDA 的 wmma::mma_sync）
                    Matmul(cL0, aL0, bL0, {{TILE_M, TILE_N, TILE_K}});
                }}

                // Step4: L0C → UB → GM（写回结果）
                LocalTensor<float> cUB = cUBBuf.Get<float>();
                DataCopy(cUB, cL0, TILE_M * TILE_N);  // L0C → UB
                DataCopy(cGm[mi*TILE_M*N + ni*TILE_N], cUB, TILE_M * TILE_N);  // UB → GM
            }}
        }}
    }}

private:
    GlobalTensor<half>  aGm, bGm;
    GlobalTensor<float> cGm;
    TPipe pipe;
    TBuf<TPosition::A1>    aL1Buf, bL1Buf;  // L1
    TBuf<TPosition::A2>    aL0Buf;           // L0A
    TBuf<TPosition::B2>    bL0Buf;           // L0B
    TBuf<TPosition::CO1>   cL0Buf;           // L0C
    TBuf<TPosition::VECCALC> cUBBuf;         // UB
    int32_t M, N, K;
}};

extern "C" __global__ __aicore__ void {op_name}_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, int32_t M, int32_t N, int32_t K)
{{
    {op_name.title().replace("_", "")}Kernel op;
    op.Init(a, b, c, M, N, K);
    op.Process();
}}
'''


def gen_reduction_kernel(op_ir: OperatorIR, tiling: TilingConfig) -> str:
    """
    生成规约算子 (softmax/layernorm/sum) 的 AscendC 代码
    规约操作需要跨 AI Core 的结果合并
    """
    return f'''/**
 * AscendC Reduction Kernel: {op_ir.name}
 * 规约操作的挑战：需要跨 AI Core 通信，通过 GM 中间结果实现
 *
 * 策略：两阶段规约
 *   Phase1: 每个 AI Core 对本地数据做局部规约 → 写入 partial_results GM
 *   Phase2: 单个 AI Core 汇总所有 partial_results
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t TILE_LENGTH = {tiling.tile_length};

extern "C" __global__ __aicore__ void {op_ir.name}_kernel(
    GM_ADDR x_gm, GM_ADDR partial_gm, GM_ADDR out_gm,
    int32_t rows, int32_t cols, int32_t phase)
{{
    // Phase 1: 每个 Core 处理部分行
    if (phase == 0) {{
        GlobalTensor<half>  xGm;
        GlobalTensor<float> partialGm;
        xGm.SetGlobalBuffer((__gm__ half*)x_gm, rows * cols);
        partialGm.SetGlobalBuffer((__gm__ float*)partial_gm, GetBlockNum() * cols);

        TPipe pipe;
        TBuf<TPosition::VECCALC> xBuf, accBuf;
        pipe.InitBuffer(xBuf,  TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(accBuf, cols * sizeof(float));

        int32_t rows_per_core = (rows + GetBlockNum() - 1) / GetBlockNum();
        int32_t start_row = GetBlockIdx() * rows_per_core;
        int32_t end_row   = min(start_row + rows_per_core, rows);

        LocalTensor<float> acc = accBuf.Get<float>();
        Duplicate(acc, 0.0f, cols);  // 初始化为 0

        for (int32_t row = start_row; row < end_row; row++) {{
            LocalTensor<half> xLocal = xBuf.Get<half>();
            DataCopy(xLocal, xGm[row * cols], cols);
            // 累加到局部结果（向量指令）
            // Add(acc, acc, xLocal, cols);  // 需要类型转换
        }}

        // 写局部规约结果
        DataCopy(partialGm[GetBlockIdx() * cols], acc, cols);
    }}
    // Phase 2: Core 0 汇总
    else if (phase == 1 && GetBlockIdx() == 0) {{
        // 读取所有局部结果，做最终规约
        // ... 实现省略
    }}
}}
'''


# ============================================================
# 主代码生成函数
# ============================================================

def generate_ascendc_kernel(
    op_ir: OperatorIR,
    ub_size_kb: int = 256,
    l0_size_kb: int = 64,
    num_cores: int = 24,
) -> dict:
    """
    根据 OperatorIR 和硬件参数，生成 AscendC 内核代码

    Returns:
        {
          "kernel_code": str,       # AscendC C++ 代码
          "tiling_config": dict,    # tiling 参数
          "build_cmd": str,         # atc 编译命令
          "host_code": str,         # Python/C++ host 调用代码
        }
    """
    tiling = compute_tiling(op_ir, ub_size_kb, l0_size_kb, num_cores)

    # 根据算子类型选择模板
    if op_ir.category == OperatorCategory.ELEMENTWISE:
        kernel_code = gen_elementwise_kernel(op_ir, tiling)
    elif op_ir.category in (OperatorCategory.MATMUL, OperatorCategory.ATTENTION):
        kernel_code = gen_matmul_kernel(op_ir, tiling)
    elif op_ir.category == OperatorCategory.REDUCTION:
        kernel_code = gen_reduction_kernel(op_ir, tiling)
    else:
        kernel_code = gen_elementwise_kernel(op_ir, tiling)  # 默认

    # 编译命令
    build_cmd = (
        f"atc --framework=5 --soc_version=Ascend910B "
        f"--input_format=ND --kernel_name={op_ir.name}_kernel "
        f"--src_file={op_ir.name}.cpp --out_file={op_ir.name}.o"
    )

    # Host 调用代码
    host_code = f'''# Python 调用昇腾算子（通过 AscendCL）
import acl
import numpy as np

def run_{op_ir.name}(x: np.ndarray) -> np.ndarray:
    # 1. 初始化 AscendCL
    acl.init()
    device_id = 0
    acl.rt.set_device(device_id)
    context, _ = acl.rt.create_context(device_id)
    stream, _ = acl.rt.create_stream()

    # 2. 分配设备内存
    x_device, _ = acl.rt.malloc(x.nbytes, acl.ACL_MEM_MALLOC_HUGE_FIRST)
    out_device, _ = acl.rt.malloc(x.nbytes, acl.ACL_MEM_MALLOC_HUGE_FIRST)

    # 3. H2D 拷贝
    acl.rt.memcpy(x_device, x.nbytes, x.ctypes.data, x.nbytes,
                  acl.ACL_MEMCPY_HOST_TO_DEVICE)

    # 4. 启动 kernel（{num_cores} 个 AI Core 并行）
    # acl.rt.launch_kernel("{op_ir.name}_kernel", {num_cores}, args, stream)
    acl.rt.synchronize_stream(stream)

    # 5. D2H 拷贝
    out = np.empty_like(x)
    acl.rt.memcpy(out.ctypes.data, out.nbytes, out_device, out.nbytes,
                  acl.ACL_MEMCPY_DEVICE_TO_HOST)
    return out
'''

    return {
        "kernel_code": kernel_code,
        "tiling_config": {
            "tile_length": tiling.tile_length,
            "tile_m": tiling.tile_m,
            "tile_n": tiling.tile_n,
            "tile_k": tiling.tile_k,
            "num_cores": tiling.num_cores,
            "double_buffer": tiling.double_buffer,
        },
        "build_cmd": build_cmd,
        "host_code": host_code,
    }
