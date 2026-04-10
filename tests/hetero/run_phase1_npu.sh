#!/bin/bash
# ────────────────────────────────────────────────────────────
# NPU Phase 1 测试脚本 — 昇腾 910B
# 直接在 ModelArts 实例上运行（不需要 Slurm）
#
# 用法:
#   bash tests/hetero/run_phase1_npu.sh              # mock 模式
#   bash tests/hetero/run_phase1_npu.sh qwen          # 用 Qwen LLM
#   bash tests/hetero/run_phase1_npu.sh qwen silu gelu # 只测指定算子
# ────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/../.."

LLM_BACKEND="${1:-mock}"
shift 2>/dev/null || true
OPS=("$@")

echo "============================================"
echo "  NPU Phase 1 — Ascend 910B"
echo "  LLM: ${LLM_BACKEND}"
echo "  Operators: ${OPS[*]:-all}"
echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# 检查 NPU 环境
echo "[1/3] Checking NPU environment..."
npu-smi info 2>/dev/null | grep -E "NPU|910" || echo "Warning: npu-smi not available"
python -c "import torch; import torch_npu; print(f'NPU available: {torch_npu.npu.is_available()}, device: {torch.npu.get_device_name(0)}')" 2>/dev/null || {
    echo "ERROR: torch_npu not available"
    exit 1
}

# 确保输出目录存在
mkdir -p output/hetero_results

# 运行测试
echo "[2/3] Running NPU tests..."
if [ ${#OPS[@]} -gt 0 ]; then
    python tests/hetero/npu_test.py --llm "${LLM_BACKEND}" --ops "${OPS[@]}"
else
    python tests/hetero/npu_test.py --llm "${LLM_BACKEND}"
fi

# 显示结果文件
echo "[3/3] Results:"
ls -la output/hetero_results/phase1_npu_*.json 2>/dev/null || echo "No result files found"
echo ""
echo "Done."
