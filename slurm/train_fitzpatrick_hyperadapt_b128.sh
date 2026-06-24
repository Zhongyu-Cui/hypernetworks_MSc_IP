#!/bin/bash
#SBATCH --partition=gpus48                                              # 48GiB (A6000/L40/L40S)：HyperAdapt 逐样本卷积核显存大，batch_size=128 在 gpus24 易 OOM，故选 gpus48 保住 bs=128 与 baseline 等 batch
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/train_fitzpatrick_hyperadapt_b128.%N.%A_%a.log
#SBATCH --time=0-16:00:00                                               # Fitzpatrick 仅 ~12.8k 图，单 epoch 远快于 MIMIC；HyperAdapt 比 baseline 慢，早停上限 30 epoch 留足余量
#SBATCH --job-name=train_fitz_hyperadapt_b128
#SBATCH --array=0-2                                                     # 3 个 seed: 42/43/44 (见 configs/fitzpatrick_baseline.yaml training.seeds)

# 等 batch 对照（gpus48 + bs=128）：与原 train_fitzpatrick_hyperadapt.sh（gpus24 + bs=32）唯一区别是
#   分区改 gpus48、batch_size 用 config 默认 128（不传 --batch_size 覆盖），lr 等训练策略完全不变。
# 目的：原 Fitzpatrick HyperAdapt 在 bs=32、baseline 在 bs=128，二者非等 batch；仿 HAM10000 用 bs=128
#   重跑翻盘的做法，把 batch 这一混淆项消掉，确认 I(Y;skin|X)≈0、HN 无收益的结论不受 batch 影响。
# checkpoint 用 --ckpt_suffix _b128 区分，避免覆盖原 bs=32 的权重（保留两份做对比）。

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_fitzpatrick_hyperadapt.py --seed "$SEED" --ckpt_suffix _b128
