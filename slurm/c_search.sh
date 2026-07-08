#!/bin/bash
# ============================================================
# C 阶段·超参搜索通用提交脚本（协议 §3 / 工作清单 C*.1–C*.4）
# ============================================================
# array 0–5 → config_index 0–5（6 配置：lr∈{3e-5,1e-4,3e-4}×wd∈{1e-4,1e-3}），
# seed 固定 42（SEEDS_SEARCH）。每方法一次提交产出 6 条子群 AUC 向量的 val 日志，
# 供 A1 Pareto 选择。
#
# 数据集 / 方法差异由 sbatch 命令行 + --export 注入，避免为每(数据集×方法)堆脚本：
#   --partition=<gpus24|gpus48>      分区（HyperAdapt bs128 用 gpus48，余 gpus24）
#   --job-name=<name>                作业名
#   --output=<logpath>               日志路径
#   --export=ALL,PY_SCRIPT=<abs.py>[,BATCH=<n>][,SWAD=1]
#     PY_SCRIPT : 训练脚本绝对路径（如 .../train_ham10000_resnet18.py）
#     BATCH     : 可选，覆盖 config 的 batch_size（等 batch 对照时留空=用 config）
#     SWAD      : 可选，=1 时加 --swad（一般搜索阶段不用，确认阶段派生 SWAD 才用）
#
# 示例（HAM ERM 搜索）：
#   sbatch --partition=gpus24 --job-name=c1_erm_search \
#          --output=/vol/biomedic2/bglocker_studproj/zc125/logs/c1_erm_search.%N.%A_%a.log \
#          --export=ALL,PY_SCRIPT=/vol/.../src/training/train_ham10000_resnet18.py \
#          slurm/c_search.sh
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00:00
# semois GPU 反复 device-handle 故障(实测 86min/epoch)，永久排除
#SBATCH --exclude=semois
#SBATCH --array=0-5

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

if [[ -z "$PY_SCRIPT" ]]; then
    echo "ERROR: 必须经 --export 传入 PY_SCRIPT=<训练脚本绝对路径>" >&2
    exit 1
fi

CONFIG_INDEX=$SLURM_ARRAY_TASK_ID   # array 0–5 直接映射到 config_index 0–5
SEARCH_SEED=42                       # SEEDS_SEARCH：搜索阶段每配置 1 seed

echo "C_SEARCH  PY_SCRIPT=$PY_SCRIPT  config_index=$CONFIG_INDEX  seed=$SEARCH_SEED  BATCH=${BATCH:-config}  SWAD=${SWAD:-0}"

python "$PY_SCRIPT" \
    --config_index "$CONFIG_INDEX" \
    --seed "$SEARCH_SEED" \
    ${BATCH:+--batch_size "$BATCH"} \
    ${SWAD:+--swad}
