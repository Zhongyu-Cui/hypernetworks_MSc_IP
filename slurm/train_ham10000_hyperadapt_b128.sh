#!/bin/bash
#SBATCH --partition=gpus48                                              # 48GiB (A6000/L40/L40S)；HyperAdapt 每层逐样本卷积核显存大，batch=128 在 gpus24(24GiB) 必 OOM，故上 gpus48
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/train_ham10000_hyperadapt_b128.%N.%A_%a.log
#SBATCH --time=0-08:00:00
#SBATCH --job-name=train_ham_hyperadapt_b128
#SBATCH --array=0-2                                                     # 3 个 seed: 42/43/44

# HyperAdapt batch=128 重跑：去掉「batch=32 vs 其它变体 128」的混淆，判定 age-HyperAdapt 的负结果
# 有多少来自小 batch 的训练噪声 vs 架构本身。除 batch_size 与分区外，与 batch-32 run 逐字一致。
# checkpoint 用 --ckpt_suffix _b128 区分，不覆盖原 batch-32 权重。
# ⚠️ batch=128 显存约为 batch-32 的 ~4×，gpus48(48GiB) 可能仍 OOM；若 OOM 退到 --batch_size 64。

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# 不传 --batch_size → 用 config 的 128（与 baseline/HyperHead/HyperFusion 对齐）
python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_ham10000_hyperadapt.py --seed "$SEED" --ckpt_suffix _b128
