#!/bin/bash
#SBATCH --partition=gpus24                                              # 单卡训练首选分区 (24GiB；ResNet-18 预训练 @ 224x224, batch_size=128)
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/train_fitzpatrick_resnet18.%N.%A_%a.log
#SBATCH --time=0-08:00:00                                               # Fitzpatrick 仅 ~12.8k 图，单 epoch 远快于 MIMIC；早停上限 30 epoch
#SBATCH --job-name=train_fitz_resnet18
#SBATCH --array=0-2                                                     # 3 个 seed: 42/43/44 (见 configs/fitzpatrick_baseline.yaml training.seeds)

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_fitzpatrick_resnet18.py --seed "$SEED"
