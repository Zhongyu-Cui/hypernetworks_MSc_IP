#!/bin/bash
#SBATCH --partition=gpus24                                          # 24GiB (RTX 3090/4090)：HyperFusion 只在单个 1x1 downsample 上展开逐样本卷积，显存接近 baseline，gpus24 足够
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/train_chexpert_hyperfusion.%N.%A_%a.log
#SBATCH --time=0-23:59:00                                           # 早停上限 30 epoch；单 epoch 与 baseline/HyperHead 同量级
#SBATCH --job-name=train_chexpert_hyperfusion
#SBATCH --array=0-2                                                 # 3 个 seed: 42/43/44 (见 configs/chexpert_baseline.yaml training.seeds)

# 显存：HyperFusion 仅在 layer4[0].downsample（1x1 conv, 256->512）上用 grouped-conv 展开
#   逐样本卷积核，复制量很小，显存占用接近 baseline / HyperHead，batch_size=128 在 gpus24 即可。

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_chexpert_hyperfusion.py --seed "$SEED"
