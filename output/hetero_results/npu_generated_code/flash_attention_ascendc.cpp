#include <kernel_operator.h>

using namespace AscendC;

// 定义常量
constexpr int32_t BLOCK_SIZE = 32; // AI Core 数量通常对应 Block 维度，但在 AscendC Kernel 中通常由启动配置决定
constexpr int32_t TILE_M = 16;     // Q 的 seq_len 分块大小 (必须 16 对齐)
constexpr int32_t TILE_N = 16;     // K/V 的 seq_len 分块大小 (必须 16 对齐)
constexpr int32_t TILE_K = 16;     // Head Dim 分块大小 (必须 16 对齐)

// 假设最大支持的 Head Dim 为 128，实际可根据模板参数调整
constexpr int32_t MAX_HEAD_DIM = 128;
constexpr int32_t BUFFER_NUM = 2;  // 双缓冲

template <typename T>
class FlashAttentionKernel {
public:
    __aicore__ inline FlashAttentionKernel() {}
    
    // 初始化函数：解析输入输出全局指针，计算分块信息
    __aicore__ inline void Init(GM_ADDR q_gm, GM_ADDR k_gm, GM_ADDR v_gm, GM_ADDR o_gm, 
                                int32_t batch, int32_t num_heads, int32_t seq_len_q, int32_t seq_len_kv, int32_t head_dim) {
        this->batch = batch;
        this->num_heads = num_heads;
        this->seq_len_q = seq_len_q;
        this->seq_len_kv = seq_len_kv;
        this->head_dim = head_dim;
        this->scale = 1.0f / sqrtf((float)head_dim);

        // 初始化 Global Tensor
        qGlobal.SetGlobalBuffer((__gm__ T*)q_gm, {batch, num_heads, seq_len_q, head_dim});
        kGlobal.SetGlobalBuffer((__gm__ T*)k_gm, {batch, num_heads, seq_len_kv, head_dim});
        vGlobal.SetGlobalBuffer((__gm__ T*)v_gm, {batch, num_heads, seq_len_kv, head_dim});
        oGlobal.SetGlobalBuffer((__gm__ T*)o_gm, {batch, num_heads, seq_len_q, head_dim});

        // 计算每个 AI Core 处理的 Q 的行数 (Tile M)
        // 简单策略：沿 Seq_Len_Q 维度分块
        total_tiles_m = (seq_len_q + TILE_M - 1) / TILE_M;
        
        // 获取当前 Core 索引，用于简单的并行划分（这里假设单核处理一个 Tile 或一组 Tile）
        // 在实际生产中，通常通过 Launch Config 的 block_idx 来映射到具体的 batch/head/seq 组合
        core_idx = GetBlockIdx();
        total_cores = GetBlockNum();
    }

    // 主处理函数
    __aicore__ inline void Process() {
        // 遍历分配给当前 Core 的任务
        // 为了简化演示，我们假设每个 Core 处理一个特定的 (Batch, Head, Seq_Start) 组合
        // 实际工程中需要更复杂的 Grid 映射逻辑
        
        int32_t work_items = batch * num_heads * total_tiles_m;
        for (int32_t idx = core_idx; idx < work_items; idx += total_cores) {
            // 解码索引
            int32_t m_idx = idx % total_tiles_m;
            int32_t head_idx = (idx / total_tiles_m) % num_heads;
            int32_t batch_idx = (idx / total_tiles_m) / num_heads;

            ProcessOneTile(batch_idx, head_idx, m_idx);
        }
    }

private:
    // 处理单个 Q 的分块 (Tile M x Head_Dim)
    __aicore__ inline void ProcessOneTile(int32_t batch_idx, int32_t head_idx, int32_t m_tile_idx) {
        // 1. 定义片上内存缓冲区 (L1/UB)
        // Q_local: [TILE_M, HEAD_DIM]
        // K_local: [TILE_N, HEAD_DIM] (转置前)
        // V_local: [TILE_N, HEAD_DIM]
        // S_local: [TILE_M, TILE_N] (Attention Score)
        // O_local: [TILE_M, HEAD_DIM] (Accumulated Output)
        // M_vec: [TILE_M] (Max values for online softmax)
        // L_vec: [TILE_M] (Sum of exponents for online softmax)
        
        TPipe pipe;
        
        // 分配 Local Tensors
        // 使用 TBuf 管理 UB 空间
        TBuf<QuePosition::VECIN> ubQueue;
        
        // 注意：AscendC 中 LocalTensor 需要明确指定形状和类型
        // 由于 Head_Dim 可能变化，这里使用动态形状或最大静态形状
        // 为了代码通用性，假设 head_dim <= MAX_HEAD_DIM 且是 16 的倍数
        
        LocalTensor<T> qLocal = ubQueue.AllocTensor<T>({TILE_M, head_dim}, FORMAT_ND);
        LocalTensor<T> kLocal = ubQueue.AllocTensor<T>({TILE_N, head_dim}, FORMAT_ND);
        LocalTensor<T> vLocal = ubQueue.AllocTensor<T>({TILE_N, head_dim}, FORMAT_ND);
        LocalTensor<float> sLocal = ubQueue.AllocTensor<float>({TILE_M, TILE_N}, FORMAT_ND); // Score in FP32 for precision
        LocalTensor<T> oAccum = ubQueue.AllocTensor<T>({TILE_M, head_dim}, FORMAT_ND); // Accumulated Output
        LocalTensor<float> mVec = ubQueue.AllocTensor<float>({TILE_M}, FORMAT_ND); // Max
        LocalTensor<float> lVec = ubQueue.AllocTensor<float>({TILE_M}, FORMAT_ND); // Sum
        LocalTensor<float> newMVec = ubQueue.AllocTensor<float>({TILE_M}, FORMAT_ND);
        LocalTensor<float> newLVec = ubQueue.AllocTensor<float>({TILE_M}, FORMAT_ND);
        LocalTensor<float> expVal = ubQueue.AllocTensor<float>({TILE_M, TILE_N}, FORMAT_ND);
        LocalTensor<T> pLocal = ubQueue.AllocTensor<T>({TILE_M, TILE_N}, FORMAT_ND); // Probability (FP16)
        LocalTensor<T> tempO = ubQueue.AllocTensor<T>({TILE_M, head_dim}, FORMAT_ND);

        // 初始化 Online Softmax 状态
        // M = -inf, L = 0
        Sets(mVec, -1e20f, TILE_M);
        Sets(lVec, 0.0f, TILE_M);
        Sets(oAccum, 0.0f, TILE_M * head_dim);

        // 加载 Q 分块: [Batch, Head, M_Start:M_End, :]
        int32_t m_start = m_tile_idx * TILE_M;
        int32_t m_actual = std::min(TILE_M, seq_len_q - m_start);
        
        // DataCopy Q from GM to L1/UB
        // 需要注意 Global Tensor 的坐标访问
        auto qSlice = qGlobal[batch_idx][head_idx].GetTensor<T>({seq_len_q, head_dim});
        auto qSub = qSlice.GetTensor<T>({m_actual, head_dim}).ReShape<TILE_M, MAX_HEAD_DIM>(); // Padding if necessary
        
        // 使用 DataCopy 搬移 Q
        // 如果 m_actual < TILE_M，需要处理边界，这里简化假设填充或只处理有效部分
        DataCopy(qLocal, qSub, m_actual * head_dim);
        
        // 如果 m_actual < TILE_M，将多余部分清零或掩码，防止计算污染
        if (m_actual < TILE_M) {
            // 设置无效行为 0 (对于 Q 来说，影响的是输出的对应行，最后写回时需注意)
            // 或者在计算 Mask 时处理。为简化，假设 Seq_Len 是 TILE_M 的倍数
        }

        // 遍历 K/V 的分块 (Tile N)
        int32_t n_tiles = (seq_len_kv + TILE_N - 1) / TILE_N;
        
        for (int32_t n_tile_idx = 0; n_tile_idx < n_tiles; ++n_tile_idx) {
            int32_t n_start = n_tile_idx * TILE_N;
            int32_t n_actual = std::min(TILE_N, seq_len_kv - n_start);

            // --- Stage 1: Load K and V ---
            auto kSlice = kGlobal[batch_idx][head_idx].GetTensor<T>({seq_len_kv, head_dim});
            auto kSub = kSlice.GetTensor<T>({n_actual, head_dim});
            
            auto vSlice = vGlobal[batch_idx][head_idx].GetTensor<T>({seq_len_kv, head_dim});
            auto vSub = vSlice.GetTensor<T>({n_actual, head_dim});

            // 搬移 K 和 V 到 Local
            // 注意：K 需要参与 MatMul(Q, K^T)，所以通常需要转置或调整 MatMul 参数
            // AscendC Mmad 支持 C = A * B, 其中 A=[M,K], B=[K,N]. 
            // Q=[M, D], K^T=[D, N]. 所以我们需要 K 的转置或者使用支持转置的 Mmad
            // 这里假设我们加载 K 为 [N, D]，然后在 Mmad 中处理转置逻辑，或者预先转置
            // DaVinci Mmad 通常要求 B 矩阵在 L0B 中是特定格式。最简单的是加载 K 并转置到 L0B。
            
            // 为了简化，我们加载 K 到 UB，然后转置拷贝到 L0B 缓冲区（如果 API 支持直接转置搬移最好，否则需 Vector 转置）
            // 此处简化：假设 DataCopy 后，使用 Mmad 的转置标志位（如果可用）或手动转置
            // AscendC Mmad: Mmad(C, A, B, M, K, N). A is L0A, B is L0B.
            // Q is [M, D]. K is [N, D]. We need Q * K^T. 
            // So A=Q[M,D], B=K^T[D,N]. 
            // 我们可以加载 K[N,D] 到 UB，然后转置存放到 L0B 对应的 Buffer。
            
            // 由于代码复杂度限制，这里展示逻辑流程：
            DataCopy(kLocal, kSub, n_actual * head_dim);
            DataCopy(vLocal, vSub, n_actual * head_dim);
            
            // --- Stage 2: Compute S = Q * K^T * Scale ---
            // S shape: [M, N]
            // 清空 S
            Sets(sLocal, 0.0f, TILE_M * TILE_N);
            
            // 矩阵乘法: S = Q * K^T
            // 需要将 KLocal [N, D] 转置为 [D, N] 放入 L0B
            // 这里调用一个辅助函数或内联汇编进行转置和 MatMul
            // 伪代码: MmadTransposeB(sLocal, qLocal, kLocal, M, D, N)
            ComputeMatMulTransposeB(sLocal, qLocal, kLocal, m_actual, head_dim, n_actual);

            // Apply Scale and Mask
            // S = S * scale
            Muls(sLocal, sLocal, scale, TILE_M * TILE_N);
            
            // Apply Causal Mask (if needed): if col > row, set to -inf
            // 这里省略 Causal Mask 的具体实现，假设是非 Causal 或外部处理
            
            // --- Stage 3: Online Softmax Update ---
            // Compute new M = max(old_M, row_max(S))
            // Compute new L = old_L * exp(old_M - new_M) + sum(exp(S - new_M))
            
            // 1. Row Max of S
            LocalTensor<float> rowMax = ubQueue.AllocTensor<float>({TILE_M}, FORMAT_ND);
            ReduceMax(rowMax, sLocal, TILE_N, TILE_M); // Reduce along N dimension
            
            // 2. Update M
            Max(newMVec, mVec, rowMax, TILE_M);
            
            // 3. Compute Exp(S - newM)
            // Broadcast newMVec to [M, N]
            LocalTensor<float> newMBroadcast = ubQueue.AllocTensor<float>({TILE_M, TILE_N}, FORMAT_ND);
            Broadcast(newMBroadcast, newMVec, TILE_N, TILE_M); // Repeat along N
            
            Sub(expVal, sLocal, newMBroadcast, TILE_M * TILE_N);
            Exp(expVal, expVal, TILE_M * TILE_N);
            
            // 4. Update L
            // sum_exp = sum(expVal, dim=N)
            LocalTensor<float> sumExp = ubQueue.AllocTensor<float>({TILE_M}, FORMAT_ND);
            ReduceSum(sumExp, expVal, TILE_N, TILE_M);
            
            // delta = exp(old_M - new_M)
            LocalTensor<float> delta = ubQueue.AllocTensor<float>({TILE_M}, FORMAT_ND);
            Sub(delta, mVec, newMVec, TILE_M);
            Exp(delta, delta, TILE_M);
            
            // new_L = old_L * delta + sum_exp
            Mul(lVec, lVec, delta, TILE_M);
            Add(lVec, lVec, sumExp, TILE_M);
            
            // 5. Update Output O
            // O_new = (O_old * diag(delta) + P * V) / diag(new_L_correction?) 
            // Standard Formula: 
            // O_new = (O_old * exp(m_old - m_new) + Softmax(S) * V) 
            // But we accumulate unnormalized O usually, then divide by L at the end?
            // Flash Attention V2 approach:
            // O_acc = O_acc * diag(d) + P * V
            // where d = exp(m_old - m_new)
            // P = exp(S - m_new) / l_new ? No, usually P is just exp(S-m_new) and we divide O by L at the very end.
            // Let's stick to: O_acc = O_acc * d + (exp(S-m_new)) * V
            // And final O = O_acc / L
            
            // Reshape expVal to P (FP16) for MatMul with V
            Casts(pLocal, expVal, TILE_M * TILE_N); // Cast FP32 -> FP16
            
            // Compute P * V
            // P: [M, N], V: [N, D] -> TempO: [M, D]
            Sets(tempO, 0.0f, TILE_M * head_dim);
            ComputeMatMul(tempO, pLocal, vLocal, m_actual, n_actual, head_dim);
            
            // Update O_Accum: O_acc = O_acc * d + TempO
            // Broadcast delta to [M, D]
            LocalTensor<T> deltaBroadcast = ubQueue.AllocTensor<T>({TILE_M, head_dim}, FORMAT_ND);
            LocalTensor<float> deltaF = ubQueue.AllocTensor<float>({TILE_M}, FORMAT_ND); // Already have delta
            // Cast delta to T
            LocalTensor<T> deltaT = ubQueue.AllocTensor<T>({TILE_M}, FORMAT_ND);
            Casts(deltaT, delta, TILE_M);
            Broadcast(deltaBroadcast, deltaT, head_dim, TILE_M);
            
            Mul(oAccum, oAccum, deltaBroadcast, TILE_M * head_dim);
            Add(oAccum, oAccum, tempO, TILE_M * head_dim);
            
            // Update M
            Copy(mVec, newMVec, TILE_M);
            
            // Free temporary tensors from queue if manual management is needed, 
            // but TBuf usually handles scope-based allocation in simple kernels or requires explicit Free
            // For simplicity in this snippet, we assume stack-like behavior or sufficient UB size.
        }
        
        // --- Finalize: Divide by L ---
        // O = O_acc / L
        // Broadcast L to [M, D]
        LocalTensor<T> lVecT = ubQueue.AllocTensor<T>({TILE_M}, FORMAT_ND);
        Casts(lVecT, lVec, TILE_M);
        
        LocalTensor<T> lBroadcast = ubQueue.AllocTensor<T>({TILE_M, head_dim}, FORMAT_ND);
        Broadcast(lBroadcast, lVecT, head_dim, TILE_M);
        
        Div(oAccum, oAccum, lBroadcast, TILE_M * head_dim);
        
        // --- Write Back to Global Memory ---
        auto oSlice = oGlobal[batch_idx][head_idx].GetTensor<T>({seq_len_q, head_dim});
        auto oSub = oSlice.GetTensor<T>({m_actual, head_dim});
        
        DataCopy(oSub, oAccum, m_actual * head_dim);
    }

    // 辅助函数: C = A * B^T
    // A: [M, K], B: [N, K] -> C: [M, N]
    __aicore__ inline void ComputeMatMulTransposeB(LocalTensor<float>& C, LocalTensor<T>& A, LocalTensor<T>& B, int32_t m, int32_t k, int32_t n) {
        // 在实际 AscendC 中，需要仔细管理 L0A/L0B/L0C 的搬运和 Mmad 指令
        // 这里使用伪代码表示核心逻辑，因为完整的 L0 管理非常繁琐且依赖具体 Shape
        
        // 1. Copy A to L0A
        // 2. Copy B to L0B (and Transpose it to [K, N] layout required by Cube)
        // 3. Mmad(C, A, B_transposed, M, K, N)
        
        // 注意：AscendC 的 Mmad 接口可能因版本而异，以下为标准调用模式示意
        // 假设存在封装好的 MmadTransposeB 或者手动处理
        
        // 由于无法在此处编写完整的汇编级 L0 管理代码，
        // 我们假设有一个高层 API 或宏来处理这个常见的 Pattern
        
        // 示例：
        // Mmad<CubeFormat::ND>(C, A, B, m, k, n, true, false); // true for Transpose B
        // 如果 API 不支持直接 Transpose，则需要先 Vector Transpose B
        
        // 这里为了代码可编译性（概念上），我们调用一个占位符
        // 在实际开发中，你需要使用 `Mmad` 并配合 `SetMatrixMode` 或手动转置
        
        // 修正：AscendC 推荐方式使用 `Mmad` 并指定转置属性
        // Mmad(C, A, B, M, K, N, transA, transB)
        // 但 L0 数据布局必须匹配。通常 B 如果是 [N,K]，转置后为 [K,N]。
        
        // 鉴于复杂性，此处仅做逻辑占位，实际需根据 `kernel_operator.h` 具体版本实现
        // 假设 Mmad 支持自动转置或预处理
         __asm__ volatile("// Matrix Multiplication with Transpose B placeholder");
    }
    
    // 辅助函数: C = A * B
    __aicore__ inline void ComputeMatMul(LocalTensor<T>& C, LocalTensor<T>& A, LocalTensor<T>& B, int32_t m, int32_t n, int32_t k) {
         __asm__ volatile("// Matrix Multiplication placeholder");
    }

private:
    GlobalTensor<T> qGlobal;
    GlobalTensor<T> kGlobal;
    GlobalTensor<T> vGlobal;
    GlobalTensor<T> oGlobal;
    
    int32_t batch;
    int32_t num_heads;
    int32_t seq_len_q;
    int32_t seq_len_kv;
    int32_t head_dim;
    float scale;
    int32_t total_tiles_m;
    int32_t core_idx;
    int32_t total_cores;
};

// Kernel 入口函数
extern "C" __global__ __aicore__ void flash_attention_kernel(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR o, 
                                                              int32_t batch, int32_t num_heads, int32_t seq_len_q, int32_t seq_len_kv, int32_t head_dim) {
    FlashAttentionKernel<half> kernel;
    kernel.Init(q, k, v, o, batch, num_heads, seq_len_q, seq_len_kv, head_dim);
    kernel.Process();
}