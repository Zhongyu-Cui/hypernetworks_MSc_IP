#!/bin/bash
# ============================================================
# 实验 P·OOD：pred-attr HyperAdapt 的 full-target 跨库评估
# ============================================================
# 规程沿用 OOD v2 预注册（docs/ood_experiment_v2_preregistration.md）：full-target 主估计量、
# 主指标 marginal worst、config 只由 source 选定。条件输入来自**源库训练的 g** 对目标库的预测
# （scripts/build_attr_predictions_cross.py），即 g+HyperAdapt 作为单一方法整体迁移。
#
# 经 --export 注入：
#   DIRECTION : m2c / c2m
#   ATTR_MODE : soft / hard / perm / const
#   TRIALS    : trial 号（缺省 0）；seed = 42 + fold + 100×trial（见 py 脚本头注的偏离说明）
#               ⚠️ **每次只传一个 trial**：sbatch --export 的值**不能含逗号**（会被截断，
#               实测 TRIALS=1,2 只跑到 trial 1，trial 2 静默丢失）。多 trial 请分多次提交。
#   WORKERS   : 可选，DataLoader workers（缺省 8；全 target 推理 I/O 密集）
#
# 示例：
#   sbatch --partition=gpus24 --job-name=pood_m2c_soft \
#          --output=/vol/.../logs/pood_m2c_soft.%N.%j.log \
#          --export=ALL,DIRECTION=m2c,ATTR_MODE=soft slurm/p_ood_pred.sh
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00:00
#SBATCH --exclude=semois

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

if [[ -z "$DIRECTION" || -z "$ATTR_MODE" ]]; then
    echo "ERROR: 必须经 --export 传入 DIRECTION 与 ATTR_MODE" >&2
    exit 1
fi

echo "=== 实验 P·OOD  direction=$DIRECTION  attr_mode=$ATTR_MODE  trials=${TRIALS:-0} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m src.training.run_ood_cxr_pred \
    --direction "$DIRECTION" --attr_mode "$ATTR_MODE" \
    --trials "${TRIALS:-0}" --num-workers "${WORKERS:-8}"
