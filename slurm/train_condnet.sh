#!/bin/bash
#SBATCH --partition=gpus48                                              # 48GiB (A6000/L40/L40S)：C-Full 逐样本 grouped-conv 铺满全 backbone，显存重，bs=128 用 gpus48 稳妥；C-Head/C-Deep 显存小、可改投 gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_condnet.%N.%A_%a.log
#SBATCH --time=0-12:00:00                                               # 单 fold × seed；C-Full 比 baseline 慢，早停上限 30 epoch 留余量
#SBATCH --job-name=train_condnet
#SBATCH --array=0-2                                                     # 默认 3 seed（plan §3「统一 3 seed」）：42/43/44
#SBATCH --exclude=semois                                               # 永久排除坏节点（SLURM 工作纪律）

# 条件化范围消融统一训练（src/training/train_condnet.py）。
# ── 用法（sbatch 时用 --export 传数据集/范围/CV/fold；SEED 由 array 索引）──
#   sbatch --export=ALL,DATASET=ham10000,LOCATION=full,CV=5,FOLD=0 slurm/train_condnet.sh
#   LOCATION ∈ {none(ERM), head(C-Head), deep(C-Deep), full(C-Full)}
#   CV=0 单-split（默认）；CV=5 时须给 FOLD∈[0,5)。
# ── canonical base state ──：train_condnet.py 由 (DATASET, FOLD, SEED) 确定性派生 base_fc_seed，
#   保证同一 (fold,seed) 下 ERM 与三 conditioning cell 的 base backbone/fc 逐元素相等。
# ── 显存 ──：CondNet 用 grouped-conv 展开逐样本卷积核（权重复制 batch 份），C-Full 铺满全
#   backbone 最重。plan §1 锁 bs=128；若 gpus48 仍紧可用 --batch_size 覆盖（但重型 HN 显存
#   对照须等 batch，改 batch 须四 cell 一致）。

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

: "${DATASET:?须经 --export 指定 DATASET（ham10000/fitzpatrick/mimic/chexpert）}"
: "${LOCATION:?须经 --export 指定 LOCATION（none/head/deep/full）}"
CV=${CV:-0}
SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

CV_ARGS=""
if [ "$CV" -ge 2 ]; then
  : "${FOLD:?CV>=2 时须经 --export 指定 FOLD}"
  CV_ARGS="--cv $CV --fold $FOLD"
fi

echo "[train_condnet] dataset=$DATASET location=$LOCATION cv=$CV fold=${FOLD:-NA} seed=$SEED"
python "$REPO_ROOT"/src/training/train_condnet.py \
  --dataset "$DATASET" --location "$LOCATION" --seed "$SEED" $CV_ARGS
