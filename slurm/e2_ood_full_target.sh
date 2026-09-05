#!/bin/bash
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=logs/e2_ood_full_target.%N.%A_%a.log
#SBATCH --time=0-16:00:00
#SBATCH --job-name=e2_ood_ft
#SBATCH --array=0-3
#SBATCH --exclude=semois,mira05,deepmedic2
# E2 M→C full-target OOD 评估：按 cell 并行（每 cell 15 个 fold×seed 模型评完整 138,644 张 CheXpert）。
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
ulimit -n 8192 2>/dev/null || true
CELLS=(condnet_erm condnet_head condnet_deep condnet_full)
CELL=${CELLS[$SLURM_ARRAY_TASK_ID]}
echo "[e2_ood_full_target] cell=$CELL"
python "$REPO_ROOT"/scripts/run_e2_ood_full_target.py --cell "$CELL"
