#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

// grad_x 输出为 float32，避免 fp16 overflow 导致训练 NaN
__global__ void rmsnorm_backward_kernel(
    const half* __restrict__ grad_out,
    const half* __restrict__ x,
    const half* __restrict__ weight,
    float* __restrict__ grad_x,       // float* 而非 half*
    float* __restrict__ grad_weight,
    int H, float eps
) {
    extern __shared__ float smem[];
    float* sx = smem;
    float* sgo = smem + H;
    int row = blockIdx.x;
    int tid = threadIdx.x;

    for (int i = tid; i < H; i += blockDim.x) {
        sx[i] = __half2float(x[row * H + i]);
        sgo[i] = __half2float(grad_out[row * H + i]);
    }
    __syncthreads();

    if (tid == 0) {
        float ss = 0.0f;
        for (int i = 0; i < H; i++) ss += sx[i] * sx[i];
        float rms_inv = rsqrtf(ss / H + eps);
        smem[2*H] = rms_inv;
        // dot = mean(go * x * w)，注意 x 是原始输入（非 x_norm）
        float dot = 0.0f;
        for (int i = 0; i < H; i++)
            dot += sgo[i] * sx[i] * __half2float(weight[i]);  // 不含 rms_inv
        smem[2*H+1] = dot / H;
    }
    __syncthreads();

    float rms_inv = smem[2*H];
    float dot_mean = smem[2*H+1];
    for (int i = tid; i < H; i += blockDim.x) {
        float wi = __half2float(weight[i]);
        float xn = sx[i] * rms_inv;
        // 正确公式: (go*w - x_norm * mean(go*x*w)) * rms_inv
        float gx = (wi * sgo[i] - xn * dot_mean) * rms_inv;
        grad_x[row * H + i] = gx;   // 直接写 float，不做 float->half 截断
        atomicAdd(&grad_weight[i], sgo[i] * xn);
    }
}

// launcher：grad_x_fp32 指向 float32 buffer
extern "C" void launch_kernel(void* grad_out, void* x, void* weight,
                               void* grad_x_fp32, void* grad_weight,
                               int N, int H, float eps) {
    int block = min(H, 256);
    int smem = (2 * H + 2) * sizeof(float);
    rmsnorm_backward_kernel<<<N, block, smem>>>(
        reinterpret_cast<const half*>(grad_out),
        reinterpret_cast<const half*>(x),
        reinterpret_cast<const half*>(weight),
        reinterpret_cast<float*>(grad_x_fp32),   // float* 输出
        reinterpret_cast<float*>(grad_weight),
        H, eps);
}