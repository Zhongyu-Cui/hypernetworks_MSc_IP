#!/bin/bash
# ============================================================
# CV-OOF OOD 跨库评估（MIMIC↔CheXpert 双向）—— GPU 推理作业
# ============================================================
# 运行 src/training/run_ood_cxr_cv.py：source cv5 逐折选定 config checkpoint → target cv5 逐折 test，
# 落盘预测到 outputs/ood_cxr/{s}2{t}/cv5/predictions/，供 cv_oof_report 池化 + 显著性。
# 2 方向 × 5 方法 × 5 折 = 50 次评估（每次 ~1/5 target），单 GPU ~1-2h。
#
# 用法：
#   sbatch --partition=gpus24 --job-name=ood_cv \
#          --output=logs/ood_cv.%N.%j.log \
#          --export=ALL,DIRECTION=both slurm/c5_ood_cv.sh
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH --exclude=semois

# ---- 环境与仓库位置（均可用环境变量覆盖，便于换机器）----------------------
#   HN_REPO_ROOT       本仓库根目录
#   HN_CONDA_ACTIVATE  conda 的 activate 脚本；文件不存在时跳过，直接用当前 python
#   HN_CONDA_ENV       conda 环境路径（配合 HN_CONDA_ACTIVATE 使用）
REPO_ROOT="${HN_REPO_ROOT:-/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP}"
CONDA_ACTIVATE="${HN_CONDA_ACTIVATE:-/vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate}"
CONDA_ENV="${HN_CONDA_ENV:-/vol/biomedic2/bglocker_studproj/zc125/envs/medimg}"
[[ -f "$CONDA_ACTIVATE" ]] && source "$CONDA_ACTIVATE" "$CONDA_ENV"
export PYTHONPATH="$REPO_ROOT"
# ---------------------------------------------------------------------------
export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"

echo "C5_OOD_CV  direction=${DIRECTION:-both}  methods=${OOD_METHODS:-全部}"
# OOD_METHODS：可选，逗号分隔的方法子集（如实验 G 只补 hyperadapt_swad）。
# ⚠️ 不传则遍历 METHODS 全集并**覆写**既有 OOD 预测（C6/v2 结论所依赖）——只补新方法时务必传。
python -m src.training.run_ood_cxr_cv --direction "${DIRECTION:-both}" \
    ${OOD_METHODS:+--methods "$OOD_METHODS"}
echo "ood_cv done rc=$?"
