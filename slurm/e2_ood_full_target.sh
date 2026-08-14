#!/bin/bash
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/e2_ood_full_target.%N.%A_%a.log
#SBATCH --time=0-16:00:00
#SBATCH --job-name=e2_ood_ft
#SBATCH --array=0-3
#SBATCH --exclude=semois,mira05,deepmedic2
# E2 M→C full-target OOD 评估：按 cell 并行（每 cell 15 个 fold×seed 模型评完整 138,644 张 CheXpert）。
source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1
ulimit -n 8192 2>/dev/null || true
CELLS=(condnet_erm condnet_head condnet_deep condnet_full)
CELL=${CELLS[$SLURM_ARRAY_TASK_ID]}
echo "[e2_ood_full_target] cell=$CELL"
python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/scripts/run_e2_ood_full_target.py --cell "$CELL"
