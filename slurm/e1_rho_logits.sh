#!/bin/bash
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/e1_rho_logits.%N.%j.log
#SBATCH --time=0-03:00:00
#SBATCH --job-name=e1_rho_logits
#SBATCH --exclude=semois,mira05
# E1 机制诊断（plan §4）：预计算「每图 × 4 个 age 取值」的 logits 表 + 逐层 ρ_l。
# 有了 logits 表，20 次属性置换退化为查表，零额外前向。
# 用法：sbatch slurm/e1_rho_logits.sh [overall|worstcase]（默认 overall，即主结果）。
CKPT="${1:-overall}"
source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1
python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/scripts/e1_rho_and_permutation.py --mode logits --checkpoint "${CKPT}"
