#!/bin/bash
#SBATCH --partition=gpus24                                  # 单卡足够，特征抽取是轻前向
#SBATCH --gres=gpu:1
#SBATCH --exclude=semois                                    # semois GPU CUDA init 故障（68961 回落 CPU），排除
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/estimate_conditional_mi_chexpert.%N.%j.log
#SBATCH --time=0-03:59:00
#SBATCH --job-name=cmi_chex

# 估计 CheXpert No Finding I(Y;A|X)，检验敏感属性信号是否弱（E0–E3）。
# 特征抽取（~139k 张图各一次前向）需 GPU；E0–E3 分析随后在缓存特征上跑（同作业）。

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate \
    /vol/biomedic2/bglocker_studproj/zc125/envs/medimg

export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
cd /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP

python scripts/estimate_conditional_mi_chexpert.py
