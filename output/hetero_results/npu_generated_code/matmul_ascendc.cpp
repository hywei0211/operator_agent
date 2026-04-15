#include "kernel_operator.h"

using namespace AscendC;

// 定义 Tile 大小，必须满足 Cube 单元对齐要求 (16字节对齐)
// M, N, K 是单次 Cube 计算处理的维度大小
// 对于 FP16，Cube 通常处理 M=16, N=16, K=16 或更大倍数
// 这里选择较大的 Tile 以提高计算密度，但需确保 L1/L0 缓冲区足够
constexpr int32_t TILE_M = 128;
constexpr int32_t TILE_N = 128;
constexpr int32_t TILE_K = 32;

// 双缓冲深度
constexpr int32_t DOUBLE_BUFFER = 2;

template <typename T>
class MatMulKernel {
public:
    __aicore__ inline MatMulKernel() {}
    
    // 初始化函数，获取输入输出全局指针和形状信息
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, 
                                int32_t batch, int32_t m, int32_t n, int32_t k) {
        this->batch = batch;
        this->M = m;
        this->N = n;
        this->K = k;
        
        // 初始化 Global Tensor
        ga_A.SetGlobalBuffer((__gm__ T*)a, batch * m * k);
        ga_B.SetGlobalBuffer((__gm__ T*)b, batch * k * n);
        ga_C.SetGlobalBuffer((__gm__ T*)c, batch * m * n);
        
        // 计算每个 AI Core 需要处理的 Batch 数量
        // 简单策略：按 Batch 维度并行，如果 Batch < AICore 数，则部分 Core 空闲
        // 更复杂的策略可以分割 M/N 维度，这里为了代码清晰，假设 Batch 足够大或仅单 Batch
        // 实际生产中通常使用 tiling 策略将 M/N/K 分块
        
        pipe.InitBuffer(inQueueA, DOUBLE_BUFFER, TILE_M * TILE_K);
        pipe.InitBuffer(inQueueB, DOUBLE_BUFFER, TILE_K * TILE_N);
        pipe.InitBuffer(outQueueC, DOUBLE_BUFFER, TILE_M * TILE_N);
        
        // 初始化 L0 缓冲区用于 Cube 计算
        // L0A: 存储 A 的分块, L0B: 存储 B 的分块, L0C: 存储累加结果
        pipe.InitBuffer(l0aA, DOUBLE_BUFFER, TILE_M * TILE_K);
        pipe.InitBuffer(l0bB, DOUBLE_BUFFER, TILE_K * TILE_N);
        pipe.InitBuffer(l0cC, DOUBLE_BUFFER, TILE_M * TILE_N);
    }

    __aicore__ inline void Process() {
        // 遍历 Batch
        for (int32_t b = 0; b < batch; ++b) {
            // 获取当前 Batch 的基地址偏移
            auto offsetA = b * M * K;
            auto offsetB = b * K * N;
            auto offsetC = b * M * N;
            
            // 遍历 M 维度
            for (int32_t mIdx = 0; mIdx < M; mIdx += TILE_M) {
                // 遍历 N 维度
                for (int32_t nIdx = 0; nIdx < N; nIdx += TILE_N) {
                    // 初始化 C 的分块为 0
                    // 由于是累加，需要先清零
                    LocalTensor<T> localC = l0cC.AllocTensor<T>();
                    Clean(localC, TILE_M * TILE_N);
                    
                    // 遍历 K 维度进行累加
                    for (int32_t kIdx = 0; kIdx < K; kIdx += TILE_K) {
                        // 双缓冲流水线加载数据
                        // Pipe 0: 加载 A 和 B
                        // Pipe 1: 计算 Mmad
                        
                        // 异步加载 A 的分块到 L1 -> L0A
                        LocalTensor<T> localA = l0aA.AllocTensor<T>();
                        DataCopy(localA, ga_A[offsetA + mIdx * K + kIdx], TILE_M * TILE_K);
                        
                        // 异步加载 B 的分块到 L1 -> L0B
                        LocalTensor<T> localB = l0bB.AllocTensor<T>();
                        DataCopy(localB, ga_B[offsetB + kIdx * N + nIdx], TILE_K * TILE_N);
                        
                        // 等待数据加载完成
                        pipe.Barrier();
                        
                        // 执行矩阵乘法: C += A @ B
                        // Mmad 指令: D = A * B + C
                        // 注意：AscendC 的 Mmad 接口可能因版本略有不同，此处使用标准形式
                        Mmad(localC, localA, localB, localC, TILE_M, TILE_K, TILE_N);
                        
                        // 释放 L0 缓冲区，允许下一轮分配
                        l0aA.FreeTensor(localA);
                        l0bB.FreeTensor(localB);
                    }
                    
                    // 将计算结果从 L0C 写回 Global Memory
                    DataCopy(ga_C[offsetC + mIdx * N + nIdx], localC, TILE_M * TILE_N);
                    
                    // 释放 L0C
                    l0cC.FreeTensor(localC);
                }
            }
        }
    }

private:
    TPipe pipe;
    
    // Global Tensors
    GlobalTensor<T> ga_A;
    GlobalTensor<T> ga_B;
    GlobalTensor<T> ga_C;
    
    // Shape info
    int32_t batch;
    int32_t M;
    int32_t N;
    int32_t K;
    
    // Queues for Double Buffering (L1/UB)
    TBuf<QuePosition::VECIN> inQueueA;
    TBuf<QuePosition::VECIN> inQueueB;
    TBuf<QuePosition::VECOUT> outQueueC;
    
    // L0 Buffers for Cube Unit
    TBuf<QuePosition::VECCALC> l0aA;
    TBuf<QuePosition::VECCALC> l0bB;
    TBuf<QuePosition::VECCALC> l0cC;
};

// 内核入口函数
extern "C" __global__ __aicore__ void matmul_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, 
                                                     int32_t batch, int32_t m, int32_t n, int32_t k) {
    MatMulKernel<half> op;
    op.Init(a, b, c, batch, m, n, k);
    op.Process();
}