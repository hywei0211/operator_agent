#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

// grad_in 输出为 float32，避免 fp16 overflow 导致训练 NaN
__global__ void silu_backward_kernel(
    const half* __restrict__ grad_out,
    const half* __restrict__ x,
    float* __restrict__ grad_in,   // float* 而非 half*
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float go = __half2float(grad_out[idx]);
        float xi = __half2float(x[idx]);
        float sig = 1.0f / (1.0f + expf(-xi));
        float gx = go * sig * (1.0f + xi * (1.0f - sig));
        grad_in[idx] = gx;   // 直接写 float，不做 float->half 截断
    }
}

// launcher：grad_in_fp32 指向 float32 buffer
extern "C" void launch_kernel(void* grad_out, void* x, void* grad_in_fp32, int N) {
    int block = 256;
    int grid = (N + block - 1) / block;
    silu_backward_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(grad_out),
        reinterpret_cast<const half*>(x),
        reinterpret_cast<float*>(grad_in_fp32),   // float* 输出
        N);
}