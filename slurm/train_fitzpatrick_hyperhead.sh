#!/bin/bash
#SBATCH --partition=gpus24                                              # 单卡训练首选 (24GiB)；HyperHead 只改 fc、显存与 image-only baseline 一致，用原 batch_size=128
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/train_fitzpatrick_hyperhead.%N.%A_%a.log
#SBATCH --time=0-12:00:00                                               # Fitzpatrick 仅 ~12.8k 图，单 epoch 快；早停上限 30 epoch 留足余量
#SBATCH --job-name=train_fitz_hyperhead
#SBATCH --array=0-2                                                     # 3 个 seed: 42/43/44 (见 configs/fitzpatrick_baseline.yaml training.seeds)

# skin 条件化 HyperHead：backbone 与 baseline 共享，仅分类头由 skin (6 肤色型) 逐样本生成。
# HyperHead 是最浅 HN（只改 fc），与 HyperFusion（单点深层）、HyperAdapt（每层低秩）构成注入深度
# 对照。注：Fitzpatrick I(Y;skin|X)≈0（作业 67913），据核心论点 HN 在此注定无收益，本作业产出
# 「信号缺失处 null result」对照。HyperHead 不展开逐样本卷积核，显存与 baseline 一致，用原 bs=128。

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_fitzpatrick_hyperhead.py --seed "$SEED"
