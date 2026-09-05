#!/bin/bash
# ============================================================
# OOD v2 · 阶段 1：独立 trial 训练（打破 seed ≡ fold）
# ============================================================
# 预注册：docs/ood_experiment_v2_preregistration.md §2（因子）、§7（阶段 1）。
#
# 现行 CV 布局把 seed 与 fold 绑死（seed = 42 + fold），使**训练随机性**与**数据划分**的方差
# 无法分离。实测同 config/seed/fold 重跑的逐折 |ΔAUC| 可达 0.003，而待检验的效应仅 0.0018–0.0019
# ⇒ 必须补独立 trial 才能判定 H2（效应是否在训练噪声之上）。
#
# **seed 编码（零破坏性）**：`seed = 42 + fold + 10 × trial`
#   trial 0 → seed 42–46（**已存在，本脚本不重跑**）
#   trial 1 → seed 52–56
#   trial 2 → seed 62–66
# run_id = <method>_<config_tag>_seed<seed> 因此天然唯一，**不改 harness/run.py、不覆盖既有产物**。
#
# array 0–9 → (fold, trial) 解码：
#   FOLD  = task % 5          （5 折患者级 GroupKFold，cv5/fold{0..4} 已就位）
#   TRIAL = task / 5 + 1      （1 或 2）
#   SEED  = 42 + FOLD + 10 × TRIAL
#
# 由 --export 注入（与 trial 0 的原始提交参数**必须逐项一致**，否则等-batch/方法条件被破坏）：
#   PY_SCRIPT    : 训练脚本路径（相对仓库根或绝对均可）
#   CONFIG_INDEX : S1 选中 config 的网格序号（见 harness/hparam_grid.py）
#   BATCH        : 可选；HyperAdapt 必须传 128（等-batch，见项目记忆：batch 不等会制造伪影）
#   SWAD         : 可选；=1 时加 --swad（仅 ERM 用，逐折派生 SWAD——两数据集 SWAD 的选中 config
#                  与 ERM 相同，故 ERM 带 --swad 即同时补齐 SWAD）
#
# 示例（MIMIC ERM + SWAD，选中 config=lr1e-04_wd1e-04 → index 2）：
#   sbatch --partition=gpus24 --job-name=oodv2_mimic_erm \
#          --output=logs/oodv2_mimic_erm.%N.%A_%a.log \
#          --export=ALL,PY_SCRIPT=src/training/train_mimic_resnet18.py,CONFIG_INDEX=2,SWAD=1 \
#          slurm/ood_trial.sh
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
# semois GPU 反复 device-handle 故障，永久排除（C 阶段纪律）
#SBATCH --exclude=semois
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
# PY_SCRIPT 允许写成相对仓库根的路径（如 src/training/train_ham10000_hyperadapt.py）
[[ -n "$PY_SCRIPT" && "$PY_SCRIPT" != /* ]] && PY_SCRIPT="$REPO_ROOT/$PY_SCRIPT"
# ---------------------------------------------------------------------------
export PYTHONUNBUFFERED=1

if [[ -z "$PY_SCRIPT" || -z "$CONFIG_INDEX" ]]; then
    echo "ERROR: 必须经 --export 传入 PY_SCRIPT 与 CONFIG_INDEX" >&2
    exit 1
fi

CV=5
FOLD=$((SLURM_ARRAY_TASK_ID % 5))
TRIAL=$((SLURM_ARRAY_TASK_ID / 5 + 1))     # 只跑 trial 1/2；trial 0 = 既有产物
SEED=$((42 + FOLD + 10 * TRIAL))

# 硬断言：trial 0 的 seed 区间（42–46）绝不可被本脚本触及，否则会覆盖既有产物
if (( SEED < 52 )); then
    echo "ERROR: SEED=$SEED 落入 trial 0 区间（42-46），会覆盖既有产物，拒绝执行" >&2
    exit 1
fi

# CUDA 预检（快速失败）：坏 GPU 节点上 torch 会把可见设备数置 0，harness 随即静默退化到 CPU——
# 单个 MIMIC run 在 CPU 上等同挂死，且日志表面看不出异常（实测 mira05 / monal05 均如此，作业
# 77943、77963 分别在两处踩到）。宁可立刻非零退出、由人换节点重投，也不要静默烧一整个 12h 时限。
# 注：推理侧 run_ood_cxr_full_target.py 早已内置同款守卫，此处是把同一纪律补到训练侧。
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "ERROR: CUDA 不可用（多为坏 GPU 节点），拒绝静默退化到 CPU；请 --exclude 该节点后重投" >&2
    exit 1
}

echo "OOD_V2_TRIAL  PY_SCRIPT=$PY_SCRIPT  config_index=$CONFIG_INDEX  fold=$FOLD  trial=$TRIAL  seed=$SEED  cv=$CV  BATCH=${BATCH:-config}  SWAD=${SWAD:-0}"

python "$PY_SCRIPT" \
    --config_index "$CONFIG_INDEX" \
    --seed "$SEED" \
    --cv "$CV" \
    --fold "$FOLD" \
    ${BATCH:+--batch_size "$BATCH"} \
    ${SWAD:+--swad}
