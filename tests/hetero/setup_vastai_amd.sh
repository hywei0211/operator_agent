#!/bin/bash
# Vast.ai AMD ROCm 环境初始化脚本
# 在 Vast.ai 实例（选择 ROCm 镜像）中执行
set -e

echo "=========================================="
echo "  Vast.ai AMD ROCm 环境配置"
echo "  适用：RX 7900 XTX / MI250 / MI300X"
echo "=========================================="

# 1. 检查 AMD GPU
echo ""
echo "[1/5] 检查 AMD GPU..."
rocm-smi --showproductname 2>/dev/null || \
    rocm-smi 2>/dev/null || \
    echo "  (rocm-smi 不可用)"

# 检查 ROCm 版本
if command -v rocminfo &>/dev/null; then
    rocminfo 2>/dev/null | grep -E "Name:|gfx" | head -6
    echo "  ✅ ROCm 环境可用"
else
    echo "  ⚠️  ROCm 未找到"
fi

hipcc --version 2>/dev/null | head -2 || echo "  hipcc 未找到"

# 2. 安装 ROCm PyTorch
echo ""
echo "[2/5] 安装 ROCm PyTorch..."
# 根据 ROCm 版本选择对应的 PyTorch
ROCM_VER=$(rocminfo 2>/dev/null | grep -m1 "ROCm" | awk '{print $NF}' | cut -d. -f1-2 || echo "6.0")
echo "  检测到 ROCm 版本: $ROCM_VER"
pip install torch --index-url "https://download.pytorch.org/whl/rocm${ROCM_VER}" -q 2>/dev/null || \
    pip install torch -q

pip install pytest pytest-asyncio -q

# 3. 验证 HIP 编译环境
echo ""
echo "[3/5] 验证 hipcc 编译..."
cat > /tmp/test_hip.cpp << 'EOF'
#include <hip/hip_runtime.h>
__global__ void test_kernel(float* x, int N) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < N) x[idx] = x[idx] * 2.0f;
}
int main() { return 0; }
EOF

if hipcc -O2 /tmp/test_hip.cpp -o /tmp/test_hip 2>/dev/null; then
    echo "  ✅ hipcc 编译测试通过"
    python3 -c "
import torch
if torch.cuda.is_available():  # ROCm 使用 cuda 接口
    print(f'  ✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'  ✅ VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
    # 检测 gfx 架构
    import subprocess
    result = subprocess.run(['rocminfo'], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'gfx' in line.lower():
            print(f'  ✅ Architecture: {line.strip()}')
            break
" 2>/dev/null
else
    echo "  ❌ hipcc 编译失败"
    echo "  尝试: sudo apt install hipcc 或检查 ROCm 安装"
fi

# 4. 更新 GPU 数据库（如果是新型号 AMD GPU）
echo ""
echo "[4/5] 检查 GPU 数据库..."
python3 -c "
import sys; sys.path.insert(0,'.')
from knowledge_base.hardware_specs.gpu_database import GPU_DATABASE
amd_gpus = {k:v for k,v in GPU_DATABASE.items() if 'amd' in str(v.vendor).lower()}
print(f'  数据库中 AMD GPU 数量: {len(amd_gpus)}')
for k in amd_gpus: print(f'    - {k}')
" 2>/dev/null || echo "  (数据库检查失败，在项目目录中运行)"

# 5. 运行指引
echo ""
echo "[5/5] 环境配置完成，运行方式："
echo ""
echo "  # 检测本机 GPU 型号并测试"
echo "  python tests/hetero/hetero_test.py \\"
echo "      --phase 2 --gpu mi300x --backend hip --llm mock"
echo ""
echo "  # 如果是 RX 7900 XTX（消费级，gfx1100）"
echo "  python tests/hetero/hetero_test.py \\"
echo "      --phase 2 --gpu rx7900xtx --backend hip --llm mock"
echo ""
echo "=========================================="
