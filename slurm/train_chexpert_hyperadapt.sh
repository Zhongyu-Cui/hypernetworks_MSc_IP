#!/bin/bash
#SBATCH --partition=gpus48                                          # 48GiB (A6000/L40)：HyperAdapt 逐样本卷积核显存大，batch_size=128 在 gpus24 易 OOM，故选 gpus48
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/train_chexpert_hyperadapt.%N.%A_%a.log
#SBATCH --time=1-12:00:00                                           # 早停上限 30 epoch；HyperAdapt 单 epoch 比 baseline 慢，留足余量
#SBATCH --job-name=train_chexpert_hyperadapt
#SBATCH --array=0-2                                                 # 3 个 seed: 42/43/44 (见 configs/chexpert_baseline.yaml training.seeds)

# ⚠️ 显存：HyperAdapt 用 grouped-conv 展开逐样本卷积核（权重复制 batch 份），显存占用
#    明显高于 baseline / HyperHead，因此选 gpus48（48GiB）以保持 batch_size=128 与 baseline 一致。
#    若 gpus48 上仍 OOM，可在 configs/chexpert_baseline.yaml 调小 training.batch_size（训练策略其余不变）。

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_chexpert_hyperadapt.py --seed "$SEED"
