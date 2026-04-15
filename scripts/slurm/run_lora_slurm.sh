#!/bin/bash
#SBATCH --job-name=operator_lora
#SBATCH --output=logs/slurm-lora-%j.out
#SBATCH --error=logs/slurm-lora-%j.err
#SBATCH --partition=fnlp-4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# ─── 代理 ───────────────────────────────────────────────────
export https_proxy=http://10.176.52.116:7890
export http_proxy=http://10.176.52.116:7890

# ─── 路径 ───────────────────────────────────────────────────
PROJECT=/remote-home1/hywei/operator_agent_2
cd ${PROJECT}

# ─── CUDA ────────────────────────────────────────────────────
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# ─── 环境变量（从 .env 加载 API 密钥）───────────────────────
if [ -f "${PROJECT}/.env" ]; then
    export $(grep -v '^#' ${PROJECT}/.env | xargs)
fi

# ─── 参数（优先使用环境变量，其次使用默认值）────────────────
MODE=${MODE:-mock}
STEPS=${STEPS:-344}        # 默认 2 epoch（344 = 175*2 条）
LLM=${LLM:-mock}
MODEL=${MODEL:-/remote-home1/share/models/Qwen/Qwen3-0.6B}

echo "=========================================="
echo "  Operator Agent — LoRA 端到端测试"
echo "  节点: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  模式: ${MODE}"
echo "  LLM:  ${LLM}"
echo "  日期: $(date)"
echo "=========================================="

mkdir -p ${PROJECT}/logs
mkdir -p ${PROJECT}/output/qwen_lora
mkdir -p ${PROJECT}/output/full_agent

# ═══════════════════════════════════════════════════════════════
# 模式路由
# ═══════════════════════════════════════════════════════════════

if [ "$MODE" = "full_agent" ]; then
    # ── 新模式：全部算子由 Agent 系统生成 + SST-2 三模式对比 ──
    echo ""
    echo ">>> Full-Agent 模式：SiLU + RMSNorm 全由 Agent 系统生成"
    echo "    LLM 后端: ${LLM}"
    echo "    三模式对比: Custom vs Baseline vs No-Finetune"
    python3 examples/full_agent_lora_train.py \
        --mode all \
        --llm ${LLM} \
        --model ${MODEL} \
        --steps ${STEPS} \
        --output-dir output/full_agent

elif [ "$MODE" = "full_agent_custom_only" ]; then
    # ── 只跑 custom 模式（节省时间）──────────────────────────
    echo ""
    echo ">>> Full-Agent Custom 模式（仅自定义算子）"
    python3 examples/full_agent_lora_train.py \
        --mode custom \
        --llm ${LLM} \
        --model ${MODEL} \
        --steps ${STEPS} \
        --output-dir output/full_agent

elif [ "$MODE" = "baseline" ]; then
    echo ""
    echo ">>> Baseline 模式：PyTorch 原生 SiLU，无自定义算子"
    python3 examples/qwen_lora_train.py \
        --baseline \
        --model ${MODEL} \
        --steps ${STEPS}

elif [ "$MODE" = "qwen" ]; then
    echo ""
    echo ">>> Qwen LLM 模式：调用 Qwen API 生成 SiLU CUDA kernel"
    python3 examples/qwen_lora_train.py \
        --llm qwen \
        --model ${MODEL} \
        --steps ${STEPS}

else
    # 默认 mock 模式（原有脚本）
    echo ""
    echo ">>> Mock 模式：快速验证框架链路（不调用 LLM API）"
    python3 examples/qwen_lora_train.py \
        --llm mock \
        --model ${MODEL} \
        --steps ${STEPS}
fi

EXIT_CODE=$?
echo ""
echo "=========================================="
echo "  任务完成，退出码: ${EXIT_CODE}"
echo "=========================================="
exit ${EXIT_CODE}
