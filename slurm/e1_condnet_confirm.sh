#!/bin/bash
# ============================================================
# E1 确认阶段：选定 config 补跑 seed 43/44（凑满 plan §3 的「统一 3 seed」）
# ============================================================
# 搜索阶段（e1_condnet_search.sh）已用 seed=42 跑满 6 config × 5 折并据**五折 validation
# marginal-worst 均值**为每个 cell 选出 config。本脚本只对**该 cell 选定的 config**补 seed 43/44，
# 与已有的 seed=42 run 合成 3 seed（seed=42 的 run 与「确认」在 config/fold/seed/代码上完全同构，
# 直接复用、不重跑——这与主协议 SEEDS_SEARCH=(42,) ⊂ SEEDS_CONFIRM=(42,...) 的既有做法一致）。
#
# **无 test 泄漏**：config 选择只用 validation（plan §3.2「pooled-OOF / test 只用于最终评估，
# 不参与任何选择」），故复用 seed=42 的 test 预测不构成泄漏。
#
# array 0–9 → (fold, seed) 解码：
#   FOLD = task / 2                （5 折）
#   SEED = 43 + (task % 2)         （补 43、44；42 由搜索阶段提供）
#
# 用法（每个 cell 一次提交，CONFIG_INDEX 取该 cell 选出的值）：
#   sbatch --partition=<gpus24|gpus48> --job-name=e1_ham_full_confirm \
#          --output=logs/e1_ham_full_confirm.%N.%A_%a.log \
#          --export=ALL,DATASET=ham10000,LOCATION=full,CONFIG_INDEX=2 slurm/e1_condnet_confirm.sh
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH --exclude=semois,mira05
#SBATCH --array=0-9

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

: "${DATASET:?须经 --export 指定 DATASET}"
: "${LOCATION:?须经 --export 指定 LOCATION（none/head/deep/full）}"
: "${CONFIG_INDEX:?须经 --export 指定该 cell 选定的 CONFIG_INDEX（0–5）}"

CV=5
FOLD=$((SLURM_ARRAY_TASK_ID / 2))
SEED=$((43 + SLURM_ARRAY_TASK_ID % 2))

echo "E1_CONFIRM dataset=$DATASET location=$LOCATION config_index=$CONFIG_INDEX fold=$FOLD seed=$SEED cv=$CV"

python "$REPO_ROOT"/src/training/train_condnet.py \
    --dataset "$DATASET" --location "$LOCATION" \
    --config_index "$CONFIG_INDEX" --seed "$SEED" --cv "$CV" --fold "$FOLD"
