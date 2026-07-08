#!/bin/bash
# ============================================================
# C 阶段·PAPILA 全-CV 超参搜索通用提交脚本（协议 §3 / 工作清单 C5.1–C5.4）
# ============================================================
# PAPILA 特例：test≈84 眼极小，`docs/papila_preprocessing_plan.md` 明确以患者级 5 折
# GroupKFold 取代单次 split。用户裁定 C5 走「全 CV：6 配置 × 5 折都跑」。
#
# array 0–29 → (config_index, fold, seed) 三元解码：
#   CONFIG_INDEX = task % 6      （6 配置 lr∈{3e-5,1e-4,3e-4}×wd∈{1e-4,1e-3}）
#   FOLD         = task / 6      （5 折 GroupKFold，cv5/fold{0..4}）
#   SEED         = 42 + FOLD     （fold↔seed 一一映射：harness run_id=(method,config,seed) 不含 fold，
#                                  5 折若共用 seed42 会互相覆盖预测/val 日志，故按折错开 seed）
# 每方法一次提交产出 6 config × 5 fold = 30 条子群 AUC 向量的 val 日志，供折聚合 Pareto（C5.5）。
# 全-CV 下选定 config 的 5 折即确认分布，C5.6 无需再训。
#
# 数据集 / 方法差异由 sbatch 命令行 + --export 注入（同 c_search.sh 约定）：
#   --partition=<gpus24|gpus48>      分区（HyperAdapt bs64 用 gpus48 等-batch，余 gpus24）
#   --job-name / --output            作业名 / 日志路径
#   --export=ALL,PY_SCRIPT=<abs.py>[,BATCH=<n>][,SWAD=1]
#     PY_SCRIPT : PAPILA 训练脚本绝对路径（train_papila_*.py，均支持 --cv/--fold）
#     BATCH     : 可选，覆盖 config 的 batch_size（HyperAdapt 等-batch 传 64）
#     SWAD      : 可选，=1 时加 --swad（仅 ERM 用，逐折派生 SWAD 基线，C5.7 不另占作业）
#
# 示例（PAPILA ERM 全-CV 搜索 + 逐折 SWAD 派生）：
#   sbatch --partition=gpus24 --job-name=c5_erm_search \
#          --output=/vol/.../logs/c5_erm_search.%N.%A_%a.log \
#          --export=ALL,PY_SCRIPT=/vol/.../src/training/train_papila_resnet18.py,SWAD=1 \
#          slurm/c_search_papila_cv.sh
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
# semois GPU 反复 device-handle 故障，永久排除
#SBATCH --exclude=semois
#SBATCH --array=0-29

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

if [[ -z "$PY_SCRIPT" ]]; then
    echo "ERROR: 必须经 --export 传入 PY_SCRIPT=<训练脚本绝对路径>" >&2
    exit 1
fi

CV=5                                     # 5 折 GroupKFold（cv5/fold{0..4} 已就位）
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID % 6))  # 0–5：超参网格 config
FOLD=$((SLURM_ARRAY_TASK_ID / 6))          # 0–4：折号
SEED=$((42 + FOLD))                        # fold↔seed 一一映射，避免 run_id 覆盖

echo "C_SEARCH_PAPILA_CV  PY_SCRIPT=$PY_SCRIPT  config_index=$CONFIG_INDEX  fold=$FOLD  seed=$SEED  cv=$CV  BATCH=${BATCH:-config}  SWAD=${SWAD:-0}"

python "$PY_SCRIPT" \
    --config_index "$CONFIG_INDEX" \
    --seed "$SEED" \
    --cv "$CV" \
    --fold "$FOLD" \
    ${BATCH:+--batch_size "$BATCH"} \
    ${SWAD:+--swad}
