#!/bin/bash
# ============================================================
# C 阶段·5-seed 确认通用提交脚本（协议 §3 / 工作清单 C*.6，含 C*.7 SWAD 派生）
# ============================================================
# array 0–4 → SEEDS_CONFIRM=(42,43,44,45,46)，config_index 固定为 Pareto 选定值。
# 每方法一次提交产出 5 seed 的确认结果，供 A2 显著性（均值±95%CI / 配对 bootstrap / DeLong）。
# ERM 确认加 SWAD=1 → 同一次训练同产 ERM 与 method=swad 两 checkpoint（C*.7 不另占作业）。
#
# 数据集 / 方法 / 选定配置由 sbatch 命令行 + --export 注入：
#   --partition=<gpus24|gpus48>      分区
#   --job-name=<name>                作业名
#   --output=<logpath>               日志路径
#   --export=ALL,PY_SCRIPT=<abs.py>,CONFIG_INDEX=<0..5>[,BATCH=<n>][,SWAD=1]
#     PY_SCRIPT    : 训练脚本绝对路径
#     CONFIG_INDEX : Pareto 选定的超参配置序号（必填）
#     BATCH        : 可选，覆盖 config 的 batch_size
#     SWAD         : 可选，=1 时加 --swad（仅 ERM 确认用，派生 SWAD 基线）
#
# 示例（HAM ERM 确认 + SWAD 派生，假设 Pareto 选定 config_index=3）：
#   sbatch --partition=gpus24 --job-name=c1_erm_confirm \
#          --output=/vol/biomedic2/bglocker_studproj/zc125/logs/c1_erm_confirm.%N.%A_%a.log \
#          --export=ALL,PY_SCRIPT=/vol/.../src/training/train_ham10000_resnet18.py,CONFIG_INDEX=3,SWAD=1 \
#          slurm/c_confirm.sh
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00:00
# semois GPU 反复 device-handle 故障(实测 86min/epoch)，永久排除
#SBATCH --exclude=semois
#SBATCH --array=0-4

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

if [[ -z "$PY_SCRIPT" || -z "$CONFIG_INDEX" ]]; then
    echo "ERROR: 必须经 --export 传入 PY_SCRIPT=<脚本> 与 CONFIG_INDEX=<Pareto 选定 0..5>" >&2
    exit 1
fi

SEEDS=(42 43 44 45 46)               # SEEDS_CONFIRM：确认阶段 5 seed
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "C_CONFIRM  PY_SCRIPT=$PY_SCRIPT  config_index=$CONFIG_INDEX  seed=$SEED  BATCH=${BATCH:-config}  SWAD=${SWAD:-0}"

python "$PY_SCRIPT" \
    --config_index "$CONFIG_INDEX" \
    --seed "$SEED" \
    ${BATCH:+--batch_size "$BATCH"} \
    ${SWAD:+--swad}
