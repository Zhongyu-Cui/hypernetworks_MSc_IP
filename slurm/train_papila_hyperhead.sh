#!/bin/bash
#SBATCH --partition=gpus24                                              # 单卡训练首选 (24GiB)；HyperHead 只改 fc、显存与 image-only baseline 一致，用原 batch_size=64
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/train_papila_hyperhead.%N.%A_%a.log
#SBATCH --time=0-06:00:00                                               # PAPILA 极小 (train≈292)，单 epoch 极快；早停上限留足余量
#SBATCH --job-name=train_papila_hyperhead
#SBATCH --array=0-2                                                     # 3 个 seed: 42/43/44 (见 configs/papila_baseline.yaml training.seeds)

# age 条件化 HyperHead：backbone 与 baseline 共享，仅分类头由 age_group (≤60/>60 = 0/1) 逐样本生成。
# age 为 PAPILA 主敏感轴 (60 分箱)。HyperHead 是最浅 HN（只改 fc），与 HyperFusion（单点深层）、
# HyperAdapt（每层低秩）构成注入深度对照。PAPILA age_group 二值全合法，无需过滤 -1。
# HyperHead 不展开逐样本卷积核，显存与 baseline 一致，用原 bs=64。

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_papila_hyperhead.py --seed "$SEED"
