#!/bin/bash
# ============================================================
# E1 敏感性实验③：训练 fc-off 的 C-Deep / C-Full（关 fc 条件化、只留 conv）
# ============================================================
# 检验：关掉 fc 条件化后，超网络是否被"逼"着把 age 条件化路由进 conv（区分「fc 挤占」vs
# 「优化/参数化瓶颈」）。**其他所有配置与 E1 的 fc-on 版完全一致**——同 config（E1 为 C-Deep/
# C-Full 选定的 lr1e-04_wd1e-04 = config_index 2）、同 5 折、同 3 seed、同 regime，唯一变量 = 关 fc。
#
# 逐 epoch 机制探针（--diagnostics）落 outputs/.../diag_logs/<run_id>.jsonl：
#   conv ρ/ρ^between（均值+逐层）、conv adapter 梯度范数（A/B 分列，裁剪前真实梯度）、权重范数。
#
# fc-off 是全新模型（E1 未训过）⇒ 须跑满 3 seed（42/43/44）。array 0–14 → (fold, seed) 解码：
#   FOLD = task / 3            （5 折）
#   SEED = 42 + task % 3       （42/43/44）
#
# 用法（每个 cell 一次提交）：
#   sbatch --job-name=e3_ham_full_nofc \
#          --output=logs/e3_ham_full_nofc.%N.%A_%a.log \
#          --export=ALL,DATASET=ham10000,LOCATION=full_nofc,CONFIG_INDEX=2 slurm/e3_condnet_nofc.sh
#   （C-Deep 版：LOCATION=deep_nofc）
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH --exclude=semois,mira05
#SBATCH --array=0-14

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
: "${LOCATION:?须经 --export 指定 LOCATION（deep_nofc/full_nofc）}"
: "${CONFIG_INDEX:?须经 --export 指定 CONFIG_INDEX（E1 选定=2）}"

CV=5
FOLD=$((SLURM_ARRAY_TASK_ID / 3))
SEED=$((42 + SLURM_ARRAY_TASK_ID % 3))

echo "E3_NOFC dataset=$DATASET location=$LOCATION config_index=$CONFIG_INDEX fold=$FOLD seed=$SEED cv=$CV"

python "$REPO_ROOT"/src/training/train_condnet.py \
    --dataset "$DATASET" --location "$LOCATION" \
    --config_index "$CONFIG_INDEX" --seed "$SEED" --cv "$CV" --fold "$FOLD" \
    --diagnostics
