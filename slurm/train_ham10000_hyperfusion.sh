#!/bin/bash
#SBATCH --partition=gpus24                                              # 单卡训练首选 (24GiB)；HyperFusion 单点 1x1 conv 注入、显存接近 baseline，用原 batch_size=128
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/train_ham10000_hyperfusion.%N.%A_%a.log
#SBATCH --time=0-08:00:00                                               # HAM 仅 ~8k 图，单 epoch 快；早停上限 30 epoch 留足余量
#SBATCH --job-name=train_ham_hyperfusion
#SBATCH --array=0-2                                                     # 3 个 seed: 42/43/44 (见 configs/ham10000_baseline.yaml training.seeds)

# age 条件化 HyperFusion：仅 layer4[0].downsample 1x1 conv 由 age (4 有效组) 逐样本生成
# (MIP additive θ_ds=θ_0+Δθ、E_L2 投影、初始 Δθ≈0)，其余层与 baseline 共享。依据 HAM 条件互
# 信息诊断 (作业 68297)——sex 轴弱/null、age 轴有可复现 I(Y;age|X)>0，故只接 age。HyperFusion 单点
# 注入显存接近 baseline，用原 batch_size=128 与 baseline 对齐（与 HyperHead 同，区别于 HyperAdapt 的 32）。

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_ham10000_hyperfusion.py --seed "$SEED"
