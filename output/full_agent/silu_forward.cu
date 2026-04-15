#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void silu_forward_kernel(const half* __restrict__ x, half* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float xi = __half2float(x[idx]);
        float v = xi / (1.0f + expf(-xi));
        out[idx] = __float2half(v);
    }
}

extern "C" void launch_kernel(void* x, void* out, int N) {
    int block = 256;
    int grid = (N + block - 1) / block;
    silu_forward_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(out),
        N);
}