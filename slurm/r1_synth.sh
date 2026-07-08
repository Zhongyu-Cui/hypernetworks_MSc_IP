#!/bin/bash
# ============================================================
# R1.3 合成信号剂量-反应·训练提交脚本（评审补强 R1）
# ============================================================
# array 0–4 → SEEDS=(42,43,44,45,46)（5-seed 确认）。METHOD / ETA / CONFIG_INDEX 由 --export 注入。
# 复用 C2 已 Pareto 选定的 MIMIC 超参配置（图像优化 regime 对 A_syn 不变，A_syn 只进 HN 分支），
# 故直接 5-seed 确认、不再逐档重搜；子群 = A_syn（dataset=mimic_synth）。
#
#   --partition=<gpus24|gpus48>   分区（HyperAdapt bs128 用 gpus48，余 gpus24）
#   --job-name=<name>             作业名
#   --output=<logpath>            日志路径（%A_%a）
#   --export=ALL,METHOD=<m>,ETA=<η>,CONFIG_INDEX=<0..5>[,BATCH=<n>][,SWAD=1]
#     METHOD       : erm / hyperhead / hyperfusion / hyperadapt
#     ETA          : 信号档翻转率（0.5 / 0.3621 / 0.2942 / 0.2096，见 R1.2 标定表）
#     CONFIG_INDEX : C2 Pareto 选定配置（ERM=5 / HyperHead=2 / HyperFusion=2 / HyperAdapt=4）
#     BATCH        : 可选，覆盖 batch_size（HyperAdapt 等 batch 用 128）
#     SWAD         : 可选，=1 时加 --swad（仅 ERM 派生 SWAD）
#
# 示例（HyperFusion，0.05 nats 档 η=0.2942，config idx2）：
#   sbatch --partition=gpus24 --job-name=r1_hf_eta0.2942 \
#          --output=/vol/biomedic2/bglocker_studproj/zc125/logs/r1_hf_eta0.2942.%N.%A_%a.log \
#          --export=ALL,METHOD=hyperfusion,ETA=0.2942,CONFIG_INDEX=2 \
#          slurm/r1_synth.sh
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00:00
#SBATCH --exclude=semois
#SBATCH --array=0-4

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

if [[ -z "$METHOD" || -z "$ETA" || -z "$CONFIG_INDEX" ]]; then
    echo "ERROR: 必须经 --export 传入 METHOD、ETA、CONFIG_INDEX" >&2
    exit 1
fi

SEEDS=(42 43 44 45 46)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "R1_SYNTH  METHOD=$METHOD  ETA=$ETA  config_index=$CONFIG_INDEX  seed=$SEED  BATCH=${BATCH:-config}  SWAD=${SWAD:-0}"

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_mimic_synth.py \
    --method "$METHOD" \
    --eta "$ETA" \
    --config_index "$CONFIG_INDEX" \
    --seed "$SEED" \
    ${BATCH:+--batch_size "$BATCH"} \
    ${SWAD:+--swad}
