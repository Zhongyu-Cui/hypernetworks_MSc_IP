#!/bin/bash
#SBATCH --partition=gpus24                                              # 纯推理（无梯度），batch128 显存远低于训练；沿用 k→k OOD 作业（71635）的分区选择
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/eaudit_m2c_full_target.%N.%A_%a.log
#SBATCH --time=0-16:00:00                                               # 每方法 5 折 × 138,644 张 = 693k 次前向；HyperAdapt/HyperFusion 较慢，留足余量
#SBATCH --job-name=eaudit_m2c_ft
#SBATCH --array=0-4                                                     # 按方法并行：erm/swad/hyperhead/hyperfusion/hyperadapt
#SBATCH --exclude=semois,mira05                                        # semois=永久排除坏节点（工作纪律）；mira05=GPU 坏（作业 72191 报 "CUDA unknown error"，torch 可见设备数=0 会静默退化 CPU）

# E-audit.2（推理侧）：M→C OOD full-target 评估。
# 每个 source(MIMIC) fold×seed 模型评**完整 CheXpert target**（138,644 张），落盘逐样本预测
# （含 patient_id 供患者级 cluster bootstrap）。历史 M→C 胜利是 legacy k→k 口径，须在此
# full-target 主口径下复核 → 判 plan §3.1 的 OOD go/no-go。
#
# ⚠️ 完整 target CSV 已在实验室机器预生成（outputs/conditioning_ablation/chexpert_full_target_test.csv），
#    故 5 个 array 任务只读不写，无并发竞争。
#
# 分析：python scripts/eaudit_m2c_full_target.py

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

# fd 上限：多个 array 任务可能落在同一节点，各带 DataLoader worker。驱动已改用
# file_system 共享策略（根治），这里再抬高软上限作双保险（首次提交 72185 曾因
# 「Too many open files」挂掉 task 1/2）。命令失败无害（部分节点不允许提权）。
ulimit -n 8192 2>/dev/null || true

METHODS=(erm swad hyperhead hyperfusion hyperadapt)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

echo "[eaudit_m2c_full_target] method=$METHOD"
python -m src.training.run_ood_cxr_full_target --method "$METHOD"
