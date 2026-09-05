#!/bin/bash
# ============================================================
# 补落盘 SWAD 系臂的 val 预测（纯推理，不训练）
# ============================================================
# 动机：`_finalize_swad` 只把 SWAD 的验证集**子群 AUC 向量**写进 val 日志，没落逐样本 npz，
# 导致 `swad` / `hyperadapt_swad` 四个库全缺 `*_val_overall.npz`（ERM/GroupDRO/三 HN 都有）。
# 阈值类指标要求「阈值在 val 上选、绝不碰评估数据」，缺 val 预测的臂就进不了表，
# 而 SWAD 是主报告里唯一给出真实收益的基线。本作业从已存在的 `*_averaged.pth` 重跑推理补齐。
#
# 每个 array task = 一个折，内部串行跑 2 个臂（swad / hyperadapt_swad），
# 每臂同时在 val（落盘）与 test（**只比对不覆盖**的自检）上推理。
#
# 用法（每库一次提交；DATASET 经 --export 注入）：
#   sbatch --job-name=swadval_ham --output=logs/swadval_ham.%N.%A_%a.log \
#          --export=ALL,DATASET=ham10000 slurm/dump_swad_val.sh
#   DATASET ∈ {ham10000, fitzpatrick, mimic, chexpert}
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
# semois 反复 device-handle 故障；mira05 会静默退回 CPU（C 阶段永久纪律）
#SBATCH --exclude=semois,mira05
#SBATCH --array=0-4

REPO_ROOT="${HN_REPO_ROOT:-/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP}"
CONDA_ACTIVATE="${HN_CONDA_ACTIVATE:-/vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate}"
CONDA_ENV="${HN_CONDA_ENV:-/vol/biomedic2/bglocker_studproj/zc125/envs/medimg}"
[[ -f "$CONDA_ACTIVATE" ]] && source "$CONDA_ACTIVATE" "$CONDA_ENV"
export PYTHONPATH="$REPO_ROOT"
export PYTHONUNBUFFERED=1

if [[ -z "$DATASET" ]]; then
    echo "ERROR: 必须经 --export 传入 DATASET=<ham10000|fitzpatrick|mimic|chexpert>" >&2
    exit 1
fi

FOLD=$SLURM_ARRAY_TASK_ID

echo "DUMP_SWAD_VAL  dataset=$DATASET  fold=$FOLD  node=$(hostname)"
python -c "import torch; print('Device:', 'cuda' if torch.cuda.is_available() else 'CPU(!!)', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

python "$REPO_ROOT/scripts/dump_swad_val_predictions.py" \
    --dataset "$DATASET" \
    --folds "$FOLD" \
    --keep-recheck \
    --overwrite
