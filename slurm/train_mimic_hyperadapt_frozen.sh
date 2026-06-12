#!/bin/bash
#SBATCH --partition=gpus48                                                 # 48GiB：HyperAdapt 逐样本卷积核显存大，与原 HyperAdapt 实验一致选 gpus48
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/train_mimic_hyperadapt_frozen.%N.%A_%a.log
#SBATCH --time=1-12:00:00                                                  # 早停上限 30 epoch；冻结 backbone 后单 epoch 仍含逐样本卷积，留足余量
#SBATCH --job-name=train_mimic_hyperadapt_frozen
#SBATCH --array=0-2                                                        # 3 个 seed: 42/43/44 (见 configs/mimic_cxr_baseline.yaml training.seeds)

# 「冻结 backbone」HyperAdapt 变体：模型/数据/优化器/早停与原 HyperAdapt 严格一致，
# 唯一区别是 freeze_backbone=True（特征提取 backbone 冻结，梯度只流入 adapter + fc）。
# 显存：仍用 grouped-conv 展开逐样本卷积核，与原 HyperAdapt 同档，故沿用 gpus48。

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_mimic_hyperadapt_frozen.py --seed "$SEED"
