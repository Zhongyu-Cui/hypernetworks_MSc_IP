#!/bin/bash
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=logs/e1_rho_logits.%N.%j.log
#SBATCH --time=0-03:00:00
#SBATCH --job-name=e1_rho_logits
#SBATCH --exclude=semois,mira05
# E1 机制诊断（plan §4）：预计算「每图 × 4 个 age 取值」的 logits 表 + 逐层 ρ_l。
# 有了 logits 表，20 次属性置换退化为查表，零额外前向。
# 用法：sbatch slurm/e1_rho_logits.sh [overall|worstcase]（默认 overall，即主结果）。
CKPT="${1:-overall}"
# ---- 环境与仓库位置（均可用环境变量覆盖，便于换机器）----------------------
#   HN_REPO_ROOT       本仓库根目录
#   HN_CONDA_ACTIVATE  conda 的 activate 脚本；文件不存在时跳过，直接用当前 python
#   HN_CONDA_ENV       conda 环境路径（配合 HN_CONDA_ACTIVATE 使用）
REPO_ROOT="${HN_REPO_ROOT:-/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP}"
CONDA_ACTIVATE="${HN_CONDA_ACTIVATE:-/vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate}"
CONDA_ENV="${HN_CONDA_ENV:-/vol/biomedic2/bglocker_studproj/zc125/envs/medimg}"
[[ -f "$CONDA_ACTIVATE" ]] && source "$CONDA_ACTIVATE" "$CONDA_ENV"
export PYTHONPATH="$REPO_ROOT"
# ---------------------------------------------------------------------------
export PYTHONUNBUFFERED=1
python "$REPO_ROOT"/scripts/e1_rho_and_permutation.py --mode logits --checkpoint "${CKPT}"
