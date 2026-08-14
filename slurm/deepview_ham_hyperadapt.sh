#!/bin/bash
#SBATCH --partition=gpus24                                              # 纯推理，单卡即可；HyperAdapt 逐样本卷积前向偏重，gpus24 足够
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/deepview_ham_hyperadapt.%N.%j.log
#SBATCH --time=0-00:30:00                                               # n=150 的 Fisher 距离 + UMAP + 背景插值，单卡约几分钟
#SBATCH --job-name=deepview_ham

# DeepView 决策边界可视化（HAM10000 age-conditioned HyperAdapt 四年龄组对比）。
# 纯推理、非训练；也可在实验室机器 GPU 上直接跑（~4 分钟）。此脚本供需要走集群时使用。
# 可选参数经环境变量传入（见默认值）：CKPT / N / GRID / ANCHOR / SEED。

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

CKPT=${CKPT:-/vol/biomedic2/bglocker_studproj/zc125/outputs/ham10000/hyperadapt_lr3e-04_wd1e-04_seed42_best_overall.pth}
N=${N:-150}
GRID=${GRID:-45}
ANCHOR=${ANCHOR:-1}
SEED=${SEED:-42}

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/scripts/deepview_ham_hyperadapt.py \
    --ckpt "$CKPT" --n_samples "$N" --grid "$GRID" --anchor_age "$ANCHOR" --seed "$SEED"
