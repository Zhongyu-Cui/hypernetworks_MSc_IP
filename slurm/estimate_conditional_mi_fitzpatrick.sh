#!/bin/bash
#SBATCH --partition=gpus24                                              # 单卡足够，特征抽取是轻前向
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/estimate_conditional_mi_fitzpatrick.%N.%j.log
#SBATCH --time=0-01:59:00
#SBATCH --job-name=cmi_fitz

# 估计 Fitzpatrick17k malignant 的 I(Y;skin|X)，模型无关地检验肤色信号强弱（E0–E3）。
# 特征抽取（~16k 张图各一次前向）需 GPU；E0–E3 分析随后在缓存特征上跑（同作业）。
# Fitzpatrick N≈16k（MIMIC ~199k 的 ~1/12），故时长上限取 2h 足够。

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate \
    /vol/biomedic2/bglocker_studproj/zc125/envs/medimg

export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
cd /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP

python scripts/estimate_conditional_mi_fitzpatrick.py
