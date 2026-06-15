#!/bin/bash
#SBATCH --partition=gpus24                                                  # 与 baseline 同分区，结构一致显存相同
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/train_mimic_resnet18_resample.%N.%A_%a.log
#SBATCH --time=1-12:00:00                                                   # 早停上限 30 epoch；留余量
#SBATCH --job-name=train_mimic_resnet18_resample
#SBATCH --array=0-2                                                         # 3 个 seed: 42/43/44

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# 平衡强度 alpha：默认 1.0（完全平衡）。可在提交时用 --export=ALL,RESAMPLE_ALPHA=0.5 覆盖跑温和档。
RESAMPLE_ALPHA=${RESAMPLE_ALPHA:-1.0}

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_mimic_resnet18.py \
    --seed "$SEED" \
    --resample_alpha "$RESAMPLE_ALPHA"
