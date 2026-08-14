#!/bin/bash
#SBATCH --partition=gpus24                                              # 单卡足够：特征抽取是轻前向（推理）
#SBATCH --gres=gpu:1
#SBATCH --exclude=semois                                                # 永久排除坏节点（slurm-workflow-prefs）
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/cvi_%x.%N.%j.log
#SBATCH --time=0-05:59:00
#SBATCH --job-name=cvi

# OOF Conditional V-information 闸门（Hewitt et al. 2021 / Xu et al. 2020）。
# 用法：sbatch --job-name=cvi_<ds> slurm/estimate_conditional_v_information.sh <dataset>
#   <dataset> ∈ {mimic_cxr, chexpert_cxr, ham10000, fitzpatrick}
#   （papila 已在实验室机器本地跑完，N 极小无需集群。）
# 阶段 A：用 cv5 的 5 折 erm baseline（seed=42+fold）抽 train/val/test 冻结特征（GPU 前向）。
#   成本 ≈ 5 × |数据集|：ham~50k / fitz~80k / chexpert~690k / mimic~1M 次前向；时长上限 6h 对 MIMIC 充裕。
# 阶段 B：在缓存特征上拟合浅 probe，算 conditional V-info（bits）+ control task（同作业，CPU/GPU）。

DATASET="${1:?用法: sbatch slurm/estimate_conditional_v_information.sh <dataset>}"

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate \
    /vol/biomedic2/bglocker_studproj/zc125/envs/medimg

export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
cd /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP

python scripts/estimate_conditional_v_information.py --dataset "${DATASET}" --n-bootstrap 2000
