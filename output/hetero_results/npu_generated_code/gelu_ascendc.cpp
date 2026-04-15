#include "kernel_operator.h"

using namespace AscendC;

// GELU 常量定义
// sqrt(2/pi) ≈ 0.7978845608028654
constexpr float SQRT_2_OVER_PI = 0.79788456f;
// 0.044715
constexpr float COEFF_CUBIC = 0.044715f;
// 0.5
constexpr float HALF = 0.5f;
// 1.0
constexpr float ONE = 1.0f;

template <typename T>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    
    // 初始化函数：解析输入输出全局指针，计算分块信息
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, int32_t totalElements) {
        this->totalElements = totalElements;
        
        // 设置全局张量
        xGm.SetGlobalBuffer((__gm__ T*)x, totalElements);
        yGm.SetGlobalBuffer((__gm__ T*)y, totalElements);
        
        // 获取 Pipe 缓冲区大小，用于分配 Local Tensor
        pipe.InitBuffer(inQueueX, 2, TILE_SIZE * sizeof(T)); // Double Buffer: 2 blocks
        pipe.InitBuffer(outQueueY, 2, TILE_SIZE * sizeof(T)); // Double Buffer: 2 blocks
        
        // 临时缓冲区用于中间计算结果 (UB)
        // 需要存储: x, x^3, inner_term, tanh_result, final_result
        // 为了简化流水线，我们在 Compute 中动态使用 UB 空间或复用 Queue 空间
        // 这里我们假设 TILE_SIZE 足够小，或者使用额外的 TBuf
        // 由于 AscendC Vector API 通常就地操作或需要额外 buffer，我们分配足够的 UB
        pipe.InitBuffer(tmpBuf1, TILE_SIZE * sizeof(T));
        pipe.InitBuffer(tmpBuf2, TILE_SIZE * sizeof(T));
        pipe.InitBuffer(tmpBuf3, TILE_SIZE * sizeof(T));
    }

    // 主处理流程：流水线控制
    __aicore__ inline void Process() {
        int32_t loopCount = totalElements / TILE_SIZE;
        
        // 预取第一个块
        CopyIn(0);
        
        for (int32_t i = 0; i < loopCount; i++) {
            // 1. 计算当前块
            Compute(i);
            
            // 2. 搬出结果 (异步)
            CopyOut(i);
            
            // 3. 搬入下一个块 (如果还有)
            if (i + 1 < loopCount) {
                CopyIn(i + 1);
            }
        }
    }

private:
    // 数据搬入 Global -> Local (UB)
    __aicore__ inline void CopyIn(int32_t index) {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopy(xLocal, xGm[index * TILE_SIZE], TILE_SIZE);
        inQueueX.EnQue(xLocal);
    }

    // 核心计算逻辑
    __aicore__ inline void Compute(int32_t index) {
        // 获取输入数据
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        
        // 分配输出和临时 Local Tensor
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<T> t1 = tmpBuf1.Get<T>();
        LocalTensor<T> t2 = tmpBuf2.Get<T>();
        LocalTensor<T> t3 = tmpBuf3.Get<T>();

        // GELU 公式: y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        
        // Step 1: 计算 x^3 -> t1
        Mul(t1, xLocal, xLocal, TILE_SIZE); // t1 = x * x
        Mul(t1, t1, xLocal, TILE_SIZE);     // t1 = x^3

        // Step 2: 计算 0.044715 * x^3 -> t2
        Sets(t2, COEFF_CUBIC, TILE_SIZE);   // t2 = 0.044715
        Mul(t2, t2, t1, TILE_SIZE);         // t2 = 0.044715 * x^3

        // Step 3: 计算 x + 0.044715 * x^3 -> t1 (复用 t1)
        Add(t1, xLocal, t2, TILE_SIZE);     // t1 = x + coeff * x^3

        // Step 4: 乘以 sqrt(2/pi) -> t2 (复用 t2)
        Sets(t2, SQRT_2_OVER_PI, TILE_SIZE);
        Mul(t2, t2, t1, TILE_SIZE);         // t2 = sqrt(2/pi) * (...)

        // Step 5: 计算 Tanh -> t3 (复用 t3)
        Tanh(t3, t2, TILE_SIZE);            // t3 = tanh(...)

        // Step 6: 计算 1 + tanh(...) -> t2 (复用 t2)
        Sets(t2, ONE, TILE_SIZE);
        Add(t2, t2, t3, TILE_SIZE);         // t2 = 1 + tanh(...)

        // Step 7: 乘以 0.5 -> t3 (复用 t3)
        Sets(t3, HALF, TILE_SIZE);
        Mul(t3, t3, t2, TILE_SIZE);         // t3 = 0.5 * (1 + tanh(...))

        // Step 8: 最终结果 x * (...) -> yLocal
        Mul(yLocal, xLocal, t3, TILE_SIZE); // y = x * result

        // 释放输入 Tensor
        inQueueX.FreeTensor(xLocal);
        
        // 入队输出 Tensor
        outQueueY.EnQue<T>(yLocal);
    }

    // 数据搬出 Local (UB) -> Global
    __aicore__ inline void CopyOut(int32_t index) {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopy(yGm[index * TILE_SIZE], yLocal, TILE_SIZE);
        outQueueY.FreeTensor(yLocal);
    }

private:
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    int32_t totalElements;
    
    // Pipeline 管理
    TPipe pipe;
    
    // 双缓冲队列
    TBuf<QuePosition::VECIN> inQueueX;
    TBuf<QuePosition::VECOUT> outQueueY;
    
    // 临时缓冲区 (用于中间变量，避免重复 Alloc/Free 开销，虽然 AscendC 中 Get 通常也是轻量级的)
    // 注意：在实际复杂算子中，可能需要更精细的 UB 管理。这里为了清晰使用独立 TBuf。
    TBuf<QuePosition::VECCALC> tmpBuf1;
    TBuf<QuePosition::VECCALC> tmpBuf2;
    TBuf<QuePosition::VECCALC> tmpBuf3;

    // Tile Size: 必须满足 16 字节对齐。FP16 为 2 字节，所以元素个数需为 8 的倍数。
    // 910B Vector 单元宽度较大，通常 256 或 512 个元素是一个较好的平衡点。
    // 这里选择 512 个 FP16 元素 (1024 字节)，保证对齐且充分利用带宽。
    static constexpr int32_t TILE_SIZE = 512;
};

// 内核入口函数
extern "C" __global__ __aicore__ void gelu_kernel(GM_ADDR x, GM_ADDR y, int32_t totalElements) {
    KernelGelu<half> op;
    op.Init(x, y, totalElements);
    op.Process();
}