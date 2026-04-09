#!/bin/bash
#SBATCH --job-name=operator_phase1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=fnlp-4090

set -euo pipefail

# =========================
# 代理设置（计算节点无外网，需要代理访问 LLM API）
# =========================
export https_proxy=http://10.176.52.116:7890
export http_proxy=http://10.176.52.116:7890

# =========================
# 用户可覆盖参数（提交时用 VAR=... sbatch ...）
# =========================
PROJECT_DIR="${PROJECT_DIR:-$HOME/operator_agent}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PHASE="${PHASE:-1}"
GPU_ID="${GPU_ID:-rtx_4090}"
BACKEND="${BACKEND:-cuda}"
LLM_BACKEND="${LLM_BACKEND:-qwen}"

# 可选：你的 conda 环境名（不需要可留空）
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

mkdir -p logs
cd "${PROJECT_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "HOSTNAME=$(hostname)"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "PHASE=${PHASE}, GPU_ID=${GPU_ID}, BACKEND=${BACKEND}, LLM=${LLM_BACKEND}"
echo "START_TIME=$(date '+%F %T')"
echo "=========================================="

# 可选：激活 conda 环境
if [[ -n "${CONDA_ENV_NAME}" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

# 读取 .env（若存在）
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# 基础诊断信息
echo "[Diag] Python: $(${PYTHON_BIN} --version)"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[Diag] nvidia-smi:"
  nvidia-smi || true
fi

echo "=========================================="
echo "开始执行 Phase ${PHASE}..."

${PYTHON_BIN} tests/hetero/hetero_test.py \
  --phase "${PHASE}" \
  --gpu "${GPU_ID}" \
  --backend "${BACKEND}" \
  --llm "${LLM_BACKEND}"

echo "=========================================="
echo "完成时间: $(date '+%F %T')"
echo "结果目录: ${PROJECT_DIR}/output/hetero_results"
