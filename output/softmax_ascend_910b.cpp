#include "kernel_operator.h"
using namespace AscendC;
class CustomKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR out, int N) {
        xGm.SetGlobalBuffer((__gm__ half*)x, N);
        outGm.SetGlobalBuffer((__gm__ half*)out, N);
        pipe.InitBuffer(inQ, 2, 256 * sizeof(half));
        pipe.InitBuffer(outQ, 2, 256 * sizeof(half));
    }
    __aicore__ inline void Process() {
        int blockId = GetBlockIdx();
        auto xLocal = inQ.AllocTensor<half>();
        DataCopy(xLocal, xGm[blockId * 256], 256);
        inQ.EnQue(xLocal);
        auto xi = inQ.DeQue<half>();
        auto yo = outQ.AllocTensor<half>();
        Muls(yo, xi, (half)2.0f, 256);
        outQ.EnQue(yo);
        auto y = outQ.DeQue<half>();
        DataCopy(outGm[blockId * 256], y, 256);
        outQ.FreeTensor(y);
        inQ.FreeTensor(xi);
    }
private:
    GlobalTensor<half> xGm, outGm;
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQ;
    TQue<QuePosition::VECOUT, 2> outQ;
};
extern "C" __global__ __aicore__ void custom_kernel(GM_ADDR x, GM_ADDR out) {
    CustomKernel op;
    op.Init(x, out, 1024);
    op.Process();
}