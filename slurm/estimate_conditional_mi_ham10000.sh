#!/bin/bash
#SBATCH --partition=gpus24                                              # 单卡足够，特征抽取是轻前向
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/estimate_conditional_mi_ham10000.%N.%j.log
#SBATCH --time=0-01:59:00
#SBATCH --job-name=cmi_ham

# 估计 HAM10000 malignant 的 I(Y;A|X)，模型无关地检验 sex/age 信号强弱（E0–E3）。
# 特征抽取（~9948 张图 × 3 seed 各一次前向）需 GPU；E0–E3 分析随后在缓存特征上跑（同作业）。
# HAM N≈9948（比 Fitz ~16k 更小），时长上限 2h 充裕。
# ⚠️ test≈995，E3 灵敏度（MDE）是结论硬约束，见脚本 docstring。

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate \
    /vol/biomedic2/bglocker_studproj/zc125/envs/medimg

export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
cd /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP

python scripts/estimate_conditional_mi_ham10000.py
