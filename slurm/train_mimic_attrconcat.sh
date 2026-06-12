#!/bin/bash
#SBATCH --partition=gpus24                                          # 24GiB (RTX 3090/4090)：AttrConcat 仅把 conv1 输入通道由 3 增至 9，显存接近 baseline，gpus24 足够
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/train_mimic_attrconcat.%N.%A_%a.log
#SBATCH --time=0-23:59:00                                           # 早停上限 30 epoch；单 epoch 与 baseline 同量级
#SBATCH --job-name=train_mimic_attrconcat
#SBATCH --array=0-2                                                 # 3 个 seed: 42/43/44 (见 configs/mimic_cxr_baseline.yaml training.seeds)

# 显存：AttrConcat 不引入超网络，结构与 baseline 几乎相同（仅 stem conv1 加宽 6 个属性通道），
#   显存占用与 baseline 一致，batch_size=128 在 gpus24 即可。

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_mimic_attrconcat.py --seed "$SEED"
