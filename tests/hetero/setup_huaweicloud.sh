#!/bin/bash
# 华为云 ModelArts Ascend 环境初始化脚本
# 在华为云 Notebook 实例（Ascend 910B）中执行
set -e

echo "=========================================="
echo "  华为云 Ascend 910B 环境配置"
echo "=========================================="

# 1. 检查昇腾环境
echo ""
echo "[1/5] 检查昇腾驱动..."
npu-smi info 2>/dev/null || echo "  (npu-smi 不可用，检查 /dev/davinci*)"
ls /dev/davinci* 2>/dev/null && echo "  ✅ Ascend 设备已挂载" || echo "  ⚠️  未找到 Ascend 设备"

# 检查 CANN 版本
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    echo "  ✅ CANN 环境已加载"
    atc --version 2>/dev/null | head -1 || echo "  atc 编译器版本检查"
else
    echo "  ⚠️  CANN 环境未找到，尝试标准路径..."
    for path in /usr/local/Ascend /home/HwHiAiUser/Ascend; do
        if [ -d "$path" ]; then
            echo "  找到 Ascend 安装目录: $path"
            find $path -name "set_env.sh" 2>/dev/null | head -3
        fi
    done
fi

# 2. 安装 Python 依赖（华为云使用 PyTorch for Ascend）
echo ""
echo "[2/5] 安装依赖..."
# 华为 Ascend 用 torch_npu 替代 torch
pip install torch_npu -q 2>/dev/null || pip install torch -q
pip install pytest pytest-asyncio -q

# 3. 验证 AscendC 编译器
echo ""
echo "[3/5] 验证 AscendC 编译器..."
cat > /tmp/test_ascendc.cpp << 'EOF'
#include "kernel_operator.h"
using namespace AscendC;
__aicore__ inline void test_kernel(__gm__ half* x, int N) {
    int block = GetBlockIdx();
}
EOF

# 注意：atc 编译器语法较复杂，此处只测试可否调用
if command -v atc &> /dev/null; then
    echo "  ✅ atc 编译器可用"
    atc --version 2>/dev/null | head -1
else
    echo "  ⚠️  atc 未在 PATH 中，检查 CANN 路径配置"
    echo "  运行: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
fi

# 4. 检测 AI Core 数量
echo ""
echo "[4/5] 检测昇腾 AI Core..."
python3 -c "
try:
    import torch
    import torch_npu
    if torch.npu.is_available():
        count = torch.npu.device_count()
        print(f'  ✅ NPU 设备数: {count}')
        for i in range(count):
            print(f'  ✅ NPU {i}: {torch.npu.get_device_name(i)}')
    else:
        print('  NPU 不可用')
except ImportError:
    print('  torch_npu 未安装，使用模拟模式')
" 2>/dev/null || echo "  torch_npu 检查失败"

# 5. 运行指引
echo ""
echo "[5/5] 环境配置完成，运行方式："
echo ""
echo "  # Phase 2: AscendC 算子编译 + 验证"
echo "  python tests/hetero/hetero_test.py \\"
echo "      --phase 2 --gpu ascend_910b --backend ascendc --llm mock"
echo ""
echo "  # 更新 knowledge_base 中的昇腾规格（如果有新型号）"
echo "  python -c \"from knowledge_base.hardware_specs.ascend_specs import ASCEND_DATABASE; print(list(ASCEND_DATABASE.keys()))\""
echo ""
echo "=========================================="
