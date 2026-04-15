#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void rmsnorm_forward_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int H, float eps
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float sum_sq = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) {
        float v = __half2float(x[row * H + i]);
        smem[i] = v;
        sum_sq += v * v;
    }
    __syncthreads();
    // Sequential reduction by thread 0
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < H; i++) total += smem[i] * smem[i];
        smem[H] = rsqrtf(total / H + eps);
    }
    __syncthreads();
    float rms_inv = smem[H];
    for (int i = tid; i < H; i += blockDim.x) {
        out[row * H + i] = __float2half(smem[i] * rms_inv * __half2float(weight[i]));
    }
}

extern "C" void launch_kernel(void* x, void* weight, void* out, int N, int H, float eps) {
    int block = min(H, 256);
    int smem = (H + 1) * sizeof(float);
    rmsnorm_forward_kernel<<<N, block, smem>>>(
        reinterpret_cast<const half*>(x),
        reinterpret_cast<const half*>(weight),
        reinterpret_cast<half*>(out),
        H, eps);
}