#!/bin/bash
# ============================================================
# E1 搜索阶段：条件化范围消融的超参搜索（6 config × 5 折，seed 固定 42）
# ============================================================
# plan §3.2「所有 cell 使用完全相同的 LR×WD 搜索空间、fold、seed 与训练预算」（防「C-Full 因优化
# 更难被误读为范围无效」）+「每个 cell 以**五折 validation marginal-worst 均值**选 config」。
# 故每个 cell 都要跑满 6 config × 5 折才能算折均、选出该 cell 的 config。
#
# **与既有 c1_ham_cv_search.sh 的关键差异**：后者用 `SEED=42+FOLD` 把折号编进 seed（因 run_id 不含
# fold），等于把 fold 与 train_seed 绑死、无法同折多 seed。本消融要求两者**解耦**（plan §1 canonical
# base state；§3「统一 3 seed」），故这里 **SEED 恒=42**，折号改由 output_dir 承载
# （train_condnet.resolve_paths → <root>/<ds>/cv5/fold{k}/），路径唯一、不会互相覆盖。
#
# array 0–29 → (config_index, fold) 解码：
#   CONFIG_INDEX = task % 6      （6 配置 lr∈{3e-5,1e-4,3e-4} × wd∈{1e-4,1e-3}）
#   FOLD         = task / 6      （5 折 lesion 级 GroupKFold，cv5/fold{0..4} 已就位）
#   SEED         = 42            （搜索固定单 seed；选定 config 后由 e1_condnet_confirm.sh 补 43/44）
#
# 用法（每个 cell 一次提交；LOCATION ∈ {none,head,deep,full}）：
#   sbatch --partition=<gpus24|gpus48> --job-name=e1_ham_full_search \
#          --output=/vol/biomedic2/bglocker_studproj/zc125/logs/e1_ham_full_search.%N.%A_%a.log \
#          --export=ALL,DATASET=ham10000,LOCATION=full slurm/e1_condnet_search.sh
#
# ⚠️ 等-batch 纪律：四个 cell 一律用 config 的 bs=128（plan §1 锁死）。**不得**只给 C-Full 降 batch
#    ——那会把「范围」与「batch」混在一起（曾被 batch=32 的 OOM 假象坑过，见记忆
#    ham10000-conditional-mi-age-signal）。显存不够就整体改投 gpus48。
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH --exclude=semois,mira05                 # semois=永久坏节点（C 阶段纪律）；mira05=GPU device-handle 故障（作业 72191）
#SBATCH --array=0-29

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

: "${DATASET:?须经 --export 指定 DATASET（ham10000/fitzpatrick/mimic/chexpert）}"
: "${LOCATION:?须经 --export 指定 LOCATION（none/head/deep/full）}"

CV=5
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID % 6))
FOLD=$((SLURM_ARRAY_TASK_ID / 6))
SEED=42

echo "E1_SEARCH dataset=$DATASET location=$LOCATION config_index=$CONFIG_INDEX fold=$FOLD seed=$SEED cv=$CV"

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_condnet.py \
    --dataset "$DATASET" --location "$LOCATION" \
    --config_index "$CONFIG_INDEX" --seed "$SEED" --cv "$CV" --fold "$FOLD"
