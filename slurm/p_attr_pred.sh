#!/bin/bash
# ============================================================
# 实验 P·子群分类器 g 的属性预测生成（特征抽取 + probe + 温度校准）
# ============================================================
# 方案：docs/predicted_attribute_hyperadapt_plan.md §3.2。纯推理任务（冻结 ImageNet 特征），
# 显存需求小；瓶颈是图像 IO（CXR 全量约 20 万张 PNG）。故不必占 gpus48。
#
# 经 --export 注入：
#   DS      : 数据集名（fitzpatrick / ham10000 / mimic / chexpert）
#   BATCH   : 可选，特征抽取 batch（缺省 128）
#   WORKERS : 可选，DataLoader workers（缺省 8）
#
# 示例：
#   sbatch --partition=gpus24 --job-name=p_attr_mimic \
#          --output=logs/p_attr_mimic.%N.%j.log \
#          --export=ALL,DS=mimic slurm/p_attr_pred.sh
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00
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

if [[ -z "$DS" ]]; then
    echo "ERROR: 必须经 --export 传入 DS=<数据集名>" >&2
    exit 1
fi

echo "=== 实验 P: 生成 g 的属性预测  dataset=$DS ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python "$REPO_ROOT/scripts/build_attr_predictions.py" \
    --dataset "$DS" --cv 5 \
    --batch_size "${BATCH:-128}" --num_workers "${WORKERS:-8}"
