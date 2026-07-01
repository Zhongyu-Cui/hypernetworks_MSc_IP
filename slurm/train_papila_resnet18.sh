#!/bin/bash
#SBATCH --partition=gpus24                                              # 单卡训练；PAPILA 极小(train≈292, 256px)，24GiB 绰绰有余
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/train_papila_resnet18.%N.%A_%a.log
#SBATCH --time=0-01:00:00                                               # 数据极小，单折 30 epoch 仅数分钟
#SBATCH --job-name=train_papila_resnet18
#SBATCH --array=0-4                                                     # 5 折患者级 GroupKFold（见 data/splits/papila/cv5/）

# 说明：本机若有 GPU 且无 sbatch，可直接本地循环跑（任务极轻量）：
#   for k in 0 1 2 3 4; do python src/training/train_papila_resnet18.py --cv 5 --fold $k --seed 42; done

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

FOLD=$SLURM_ARRAY_TASK_ID

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_papila_resnet18.py \
    --cv 5 --fold "$FOLD" --seed 42
