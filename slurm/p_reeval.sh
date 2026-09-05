#!/bin/bash
# ============================================================
# 实验 P·从已训 checkpoint 重新评估并落盘预测（不重训）
# ============================================================
# 用于训练成功但收尾落盘失败的情形（见 scripts/p_reeval_from_checkpoints.py 头注）。
# array 0–4 → fold；每个任务处理该折的全部四臂（soft/hard/perm/const）。
#
# 经 --export 注入：
#   DS    : 数据集名（mimic / chexpert / ham10000 / fitzpatrick / papila）
#   MODES : 可选，空格分隔的模式子集（缺省全部四臂）
#
# 示例：
#   sbatch --partition=gpus48 --job-name=p_reeval_mimic \
#          --output=logs/p_reeval_mimic.%N.%A_%a.log \
#          --export=ALL,DS=mimic slurm/p_reeval.sh
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH --exclude=semois
#SBATCH --array=0-4

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

if [[ -z "$DS" ]]; then
    echo "ERROR: 必须经 --export 传入 DS=<数据集名>" >&2
    exit 1
fi

FOLD=$SLURM_ARRAY_TASK_ID
echo "=== 实验 P·重评估  dataset=$DS  fold=$FOLD ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CMD=(python "$REPO_ROOT/scripts/p_reeval_from_checkpoints.py" --dataset "$DS" --fold "$FOLD")
if [[ -n "$MODES" ]]; then
    CMD+=(--modes $MODES)
fi
echo "CMD: ${CMD[*]}"
"${CMD[@]}"
