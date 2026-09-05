#!/bin/bash
# ============================================================
# 路线 A·HAM10000 5 折 GroupKFold OOF 重训（worst-group 评估端功效补强）
# ============================================================
# 动机：HAM 单次 80/10/10 的 test 仅 995 张，worst-group 落 joint 小格（Female|80+ n_pos=4 等）
# 进噪声地板，Overall 侧 seed-ensemble 救不了公平（评审 R4.1/R4.2）。改跑 5 折 lesion 级 GroupKFold、
# 5 折 disjoint test 池化成全 9948 张 OOF → 少数子群评估 n 放大约 5–10×，把 worst-group 从评估-n 地板抬起。
#
# 算力中性设计：不重做 6-config 超参搜索，**复用 C1.5 已选定的 Pareto 配置**，每方法只跑 5 折
# （= 原 5-seed 同 run 数），只是把「同 split 多 seed」重排成「5 折 disjoint test」。
#
# array 0–4 → (fold, seed) 二元解码：
#   FOLD = SLURM_ARRAY_TASK_ID   （0–4，cv5/fold{0..4}）
#   SEED = 42 + FOLD             （fold↔seed 一一映射：run_id=(method,config,seed) 不含 fold，
#                                  5 折共用 seed 会互相覆盖预测；CV 产物另写 outputs/ham10000/cv5/）
#
# CONFIG_INDEX（C1.5 Pareto 选定，经 --export 注入）：
#   ERM idx2 (lr1e-04_wd1e-04) / HyperHead idx3 (lr1e-04_wd1e-03)
#   HyperFusion idx5 (lr3e-04_wd1e-03) / HyperAdapt idx4 (lr3e-04_wd1e-04, bs128 等-batch)
#
# 数据集/方法差异由 sbatch 命令行 + --export 注入（同 c_search_papila_cv.sh 约定）：
#   --partition=<gpus24|gpus48>   HyperAdapt bs128 用 gpus48 等-batch，余 gpus24
#   --job-name / --output         作业名 / 日志路径
#   --export=ALL,PY_SCRIPT=<abs.py>,CONFIG_INDEX=<0-5>[,BATCH=<n>][,SWAD=1][,FREEZE=1]
#     PY_SCRIPT    : HAM 训练脚本绝对路径（train_ham10000_*.py，均支持 --cv/--fold）
#     CONFIG_INDEX : C1.5 Pareto 选定配置序号（必填）
#     BATCH        : 可选，覆盖 config 的 batch_size（HyperAdapt 等-batch 传 128）
#     SWAD         : 可选，=1 时加 --swad（仅 ERM 用，逐折派生 SWAD 基线）
#     FREEZE       : 可选，=1 时加 --freeze_backbone（冻结 regime 实验 F：冻结 ImageNet 预训练
#                    backbone、只训 adapter+fc，对齐 HyperAdapt 论文；method 记为 *_frozen，与
#                    全微调结果分开存放。复用同一 Pareto config 保证仅「freeze」单变量差异）
#
# 示例（ERM 全-CV + 逐折 SWAD 派生）：
#   sbatch --partition=gpus24 --job-name=a_ham_erm_cv \
#          --output=logs/a_ham_erm_cv.%N.%A_%a.log \
#          --export=ALL,PY_SCRIPT=src/training/train_ham10000_resnet18.py,CONFIG_INDEX=2,SWAD=1 \
#          slurm/c1_ham_cv_oof.sh
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
# semois GPU 反复 device-handle 故障，永久排除（同 C 阶段纪律）
#SBATCH --exclude=semois
#SBATCH --array=0-4

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

# PY_SCRIPT 允许写成相对仓库根的路径（如 src/training/train_ham10000_hyperadapt.py）
[[ -n "$PY_SCRIPT" && "$PY_SCRIPT" != /* ]] && PY_SCRIPT="$REPO_ROOT/$PY_SCRIPT"
if [[ -z "$PY_SCRIPT" ]]; then
    echo "ERROR: 必须经 --export 传入 PY_SCRIPT=<训练脚本路径，相对仓库根或绝对均可>" >&2
    exit 1
fi
if [[ -z "$CONFIG_INDEX" ]]; then
    echo "ERROR: 必须经 --export 传入 CONFIG_INDEX=<C1.5 Pareto 选定配置序号 0-5>" >&2
    exit 1
fi

CV=5                                  # 5 折 lesion 级 GroupKFold（cv5/fold{0..4}）
FOLD=$SLURM_ARRAY_TASK_ID             # 0–4：折号
SEED=$((42 + FOLD))                   # fold↔seed 一一映射，避免 run_id 覆盖

echo "A_HAM_CV_OOF  PY_SCRIPT=$PY_SCRIPT  config_index=$CONFIG_INDEX  fold=$FOLD  seed=$SEED  cv=$CV  BATCH=${BATCH:-config}  SWAD=${SWAD:-0}  FREEZE=${FREEZE:-0}"

python "$PY_SCRIPT" \
    --config_index "$CONFIG_INDEX" \
    --seed "$SEED" \
    --cv "$CV" \
    --fold "$FOLD" \
    ${BATCH:+--batch_size "$BATCH"} \
    ${SWAD:+--swad} \
    ${FREEZE:+--freeze_backbone}
