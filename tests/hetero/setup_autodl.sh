#!/bin/bash
# AutoDL NVIDIA 环境初始化脚本
# 在 AutoDL 机器上执行：bash setup_autodl.sh
set -e

echo "=========================================="
echo "  AutoDL NVIDIA 环境配置"
echo "  适用于：RTX 4090 / RTX 3090 / vGPU 系列"
echo "=========================================="

# 1. 检查 GPU
echo ""
echo "[1/5] 检查 GPU 环境..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvcc --version | head -1

# 2. 安装依赖
echo ""
echo "[2/5] 安装 Python 依赖..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
pip install flash-attn --no-build-isolation -q 2>/dev/null || echo "  flash-attn 安装失败（可选，跳过）"
pip install pytest pytest-asyncio aiohttp beautifulsoup4 -q

# 3. 克隆/更新代码
echo ""
echo "[3/5] 准备项目代码..."
if [ ! -d "operator_agent" ]; then
    # 如果是全新机器，从本地打包上传
    echo "  请先上传项目代码到此目录，或使用 git clone"
else
    echo "  项目代码已存在"
fi

cd operator_agent 2>/dev/null || true

# 4. 验证 CUDA 编译环境
echo ""
echo "[4/5] 验证 CUDA 编译环境..."
cat > /tmp/test_nvcc.cu << 'EOF'
#include <cuda_runtime.h>
__global__ void test_kernel(float* x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) x[idx] = x[idx] * 2.0f;
}
int main() { return 0; }
EOF

if nvcc -O2 /tmp/test_nvcc.cu -o /tmp/test_nvcc 2>/dev/null; then
    echo "  ✅ nvcc 编译测试通过"
    # 探测 SM 版本
    python3 -c "
import torch
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability()
    print(f'  ✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'  ✅ Compute Capability: sm_{cc[0]}{cc[1]}')
    print(f'  ✅ VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
"
else
    echo "  ❌ nvcc 编译失败"
fi

# 5. 运行 Phase 1 测试
echo ""
echo "[5/5] 环境配置完成，运行方式："
echo ""
echo "  # Phase 1: 编译 + 数值验证"
echo "  python tests/hetero/hetero_test.py --phase 1 --gpu rtx_4090 --llm mock"
echo ""
echo "  # 如果有 OpenAI API Key（生成质量更好）："
echo "  export OPENAI_API_KEY=sk-..."
echo "  python tests/hetero/hetero_test.py --phase 1 --gpu rtx_4090 --llm openai"
echo ""
echo "=========================================="
