#!/bin/bash
#SBATCH --partition=gpus24                                              # 单卡训练首选 (24GiB)；HyperFusion 仅单点注入 (layer4[0].downsample)，显存接近 baseline，用原 batch_size=128
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/train_fitzpatrick_hyperfusion.%N.%A_%a.log
#SBATCH --time=0-12:00:00                                               # Fitzpatrick 仅 ~12.8k 图，单 epoch 快；早停上限 30 epoch 留足余量
#SBATCH --job-name=train_fitz_hyperfusion
#SBATCH --array=0-2                                                     # 3 个 seed: 42/43/44 (见 configs/fitzpatrick_baseline.yaml training.seeds)

# skin 条件化 HyperFusion：仅在 layer4[0].downsample 处由 skin (6 肤色型) 逐样本调制卷积权重
# (MIP additive θ=θ₀+h(γ)、E_L2 投影、Δθ≈0)，其余层与 baseline 共享。HyperFusion 是中层单点 HN，
# 介入深度介于 HyperHead（最浅）与 HyperAdapt（最遍布）之间。注：Fitzpatrick I(Y;skin|X)≈0
# （作业 67913），据核心论点 HN 在此注定无收益，本作业产出「信号缺失处 null result」对照。
# 单点注入显存近 baseline，用原 bs=128。

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_fitzpatrick_hyperfusion.py --seed "$SEED"
