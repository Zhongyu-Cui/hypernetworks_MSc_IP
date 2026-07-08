#!/bin/bash
#SBATCH --partition=gpus24                                              # 24GiB (RTX 3090/4090) 单卡首选；HyperAdapt 逐样本卷积核显存大，配合 --batch_size 32 控制显存
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/train_papila_hyperadapt.%N.%A_%a.log
#SBATCH --time=0-06:00:00                                               # PAPILA 极小 (train≈292)，单 epoch 极快；HyperAdapt 比 baseline 慢，早停上限留足余量
#SBATCH --job-name=train_papila_hyperadapt
#SBATCH --array=0-2                                                     # 3 个 seed: 42/43/44 (见 configs/papila_baseline.yaml training.seeds)

# age 条件化 HyperAdapt：backbone 每 conv 通道级乘性调制 + fc 加性低秩更新，均由 age_group
# (≤60/>60 = 0/1) 逐样本生成 (Δθ≈0 初始等价 baseline)。age 为 PAPILA 主敏感轴 (60 分箱)。
# HyperAdapt 是最遍布深层 HN（除 stem 外每层），与 HyperFusion（单点深层）、HyperHead（最浅）
# 构成注入深度对照。PAPILA age_group 二值全合法，无需过滤 -1。
# ⚠️ 显存：HyperAdapt 用 grouped-conv 展开逐样本卷积核（权重复制 batch 份），显存明显高于
#    baseline，只取决于 batch_size 与分辨率。为放进 gpus24，用 --batch_size 32 覆盖 config 的 64
#    （仅改 batch_size，lr 等训练策略不变、仍与 baseline 对齐）。若仍 OOM 可继续调小。

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_papila_hyperadapt.py --seed "$SEED" --batch_size 32
