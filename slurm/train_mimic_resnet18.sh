#!/bin/bash
#SBATCH --partition=gpus24                                          # 单卡训练首选分区 (24GiB；ResNet-18 预训练 @ 224x224, batch_size=128)
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/train_mimic_resnet18.%N.%A_%a.log
#SBATCH --time=1-12:00:00                                           # 早停上限 30 epoch；留出余量覆盖单 epoch 用时
#SBATCH --job-name=train_mimic_resnet18
#SBATCH --array=0-2                                                 # 3 个 seed: 42/43/44 (见 configs/mimic_cxr_baseline.yaml training.seeds)

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_mimic_resnet18.py --seed "$SEED"
