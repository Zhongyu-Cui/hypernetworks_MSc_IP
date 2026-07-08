#!/bin/bash
#SBATCH --partition=gpus24                                  # 单卡；在缓存特征上跑凸 logistic，GPU 加速 LBFGS
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/calibrate_synthetic_signal.%N.%j.log
#SBATCH --time=0-03:59:00
#SBATCH --job-name=synth_calib
#SBATCH --exclude=semois                                    # 永久排除坏节点（见项目记忆 slurm-workflow-prefs）

# R1.2：把合成属性翻转率 η 标定到目标 I(Y;A_syn|X) 档位 {0,0.02,0.05,0.10} nats。
# 复用 MIMIC baseline 冻结特征缓存（outputs/mimic_cxr/cmi_cache，无需重抽特征），
# 扫 η → 建 η↔实测 I 标定表 → 对各档插值 η* 并复核。

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate \
    /vol/biomedic2/bglocker_studproj/zc125/envs/medimg

export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
cd /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP

python scripts/calibrate_synthetic_signal.py
