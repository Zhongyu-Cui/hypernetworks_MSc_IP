#!/bin/bash
#SBATCH --partition=gpus24                                              # 24GiB (RTX 3090/4090) 单卡首选；HyperAdapt 逐样本卷积核显存大，故配合 --batch_size 32（见下）控制显存
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/train_fitzpatrick_hyperadapt.%N.%A_%a.log
#SBATCH --time=0-16:00:00                                               # Fitzpatrick 仅 ~12.8k 图，单 epoch 远快于 MIMIC；HyperAdapt 比 baseline 慢，早停上限 30 epoch 留足余量
#SBATCH --job-name=train_fitz_hyperadapt
#SBATCH --array=0-2                                                     # 3 个 seed: 42/43/44 (见 configs/fitzpatrick_baseline.yaml training.seeds)

# ⚠️ 显存：HyperAdapt 用 grouped-conv 展开逐样本卷积核（权重复制 batch 份），显存占用
#    明显高于 image-only baseline，且只取决于 batch_size 与分辨率（与数据量无关）。
#    为放进 gpus24 (24GiB)，用 --batch_size 32 覆盖 config 的 128（仅改 batch_size，lr 等
#    训练策略不变、仍与 baseline 对齐）。若仍 OOM 可继续调小（16）；若想用原 bs=128 则改投 gpus48。

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_fitzpatrick_hyperadapt.py --seed "$SEED" --batch_size 32
