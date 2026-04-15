#!/bin/bash
#SBATCH --job-name=test_qwen
#SBATCH --output=logs/slurm-test-%j.out
#SBATCH --error=logs/slurm-test-%j.err
#SBATCH --partition=fnlp-4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00

# 设置代理（只用 HTTP 代理，不用 socks5，避免缺 socksio 包）
export https_proxy=http://10.176.52.116:7890
export http_proxy=http://10.176.52.116:7890

cd /remote-home1/hywei/operator_agent
python test_qwen.py
