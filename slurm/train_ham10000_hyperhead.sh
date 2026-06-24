#!/bin/bash
#SBATCH --partition=gpus24                                              # 单卡训练首选 (24GiB)；HyperHead 只改 fc、显存与 image-only baseline 一致，用原 batch_size=128
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/train_ham10000_hyperhead.%N.%A_%a.log
#SBATCH --time=0-08:00:00                                               # HAM 仅 ~8k 图，单 epoch 快；早停上限 30 epoch 留足余量
#SBATCH --job-name=train_ham_hyperhead
#SBATCH --array=0-2                                                     # 3 个 seed: 42/43/44 (见 configs/ham10000_baseline.yaml training.seeds)

# age 条件化 HyperHead：backbone 与 baseline 共享，仅分类头由 age (4 有效组) 逐样本生成。
# 依据 HAM 条件互信息诊断 (作业 68297)——sex 轴信号弱/null、age 轴读出可复现 I(Y;age|X)>0；
# 且 HyperHead「共享 φ(X) 上按 age 生成逐样本线性头」≈诊断 E1 的 int 模型，是与该信号最对口的
# 架构。HyperHead 不展开逐样本卷积核，显存与 baseline 一致，故用原 batch_size=128（与 baseline 对齐）。

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_ham10000_hyperhead.py --seed "$SEED"
