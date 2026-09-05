#!/bin/bash
# ============================================================
# MIMIC-CXR 全-CV 超参搜索（6 config × 5 折）——CV-OOF 口径铺开（方案 R-MIMIC，最后最贵）
# ============================================================
# 对齐 slurm/c1_ham_cv_search.sh / c3_chexpert_cv_search.sh：把 MIMIC 从「master 自带单-split」升级为
# 「6 config × 5 折」患者级 GroupKFold CV 搜索，使 config 选择走折均 marginal-worst
# （scripts/select_config_cv.py），评估走 pooled-OOF（n≈199,356，scripts/cv_oof_report.py）。
# 敏感属性 = Sex+Race+Age（14 子群）；CV 折按 patient_id 分组（cv5/fold{0..4} 由
# scripts/build_mimic_splits_nofinding.py --cv 5 生成，已断言并集=199356、5 折 test disjoint、无患者跨折）。
#
# ⚠️ 成本最高：MIMIC ~90 min/run × 6 config × 5 折 × 4 方法 = 120 run ≈ 180 GPU-时。
#    如需避免占满配额，可分方法/分批提交（本脚本一次一方法 = array 0–29）。
#
# array 0–29 → (config_index, fold, seed) 三元解码（同 HAM/PAPILA/Fitz/CheXpert 约定）：
#   CONFIG_INDEX = task % 6      （6 配置 lr∈{3e-5,1e-4,3e-4}×wd∈{1e-4,1e-3}）
#   FOLD         = task / 6      （5 折患者级 GroupKFold，cv5/fold{0..4} 已就位）
#   SEED         = 42 + FOLD     （fold↔seed 一一映射：run_id 不含 fold，5 折共用 seed 会覆盖预测）
# 每方法一次提交产出 6 config × 5 fold = 30 条 val 日志 + OOF test 预测，写 outputs/mimic_cxr/cv5/。
#
# 数据集/方法差异由 sbatch 命令行 + --export 注入：
#   --partition=<gpus24|gpus48>   HyperAdapt 等-batch bs128 用 gpus48，余 gpus24
#   --job-name / --output         作业名 / 日志路径
#   --export=ALL,PY_SCRIPT=<abs.py>[,BATCH=<n>][,SWAD=1]
#     PY_SCRIPT : MIMIC 训练脚本绝对路径（train_mimic_*.py，均支持 --cv/--fold/--config_index）
#     BATCH     : 可选，覆盖 config 的 batch_size（HyperAdapt 等-batch 传 128）
#     SWAD      : 可选，=1 时加 --swad（仅 ERM 用，逐折派生 SWAD 基线）
#
# 示例（MIMIC ERM 全-CV 搜索 + 逐折 SWAD 派生）：
#   sbatch --partition=gpus24 --job-name=mimic_erm_cvsearch \
#          --output=logs/mimic_erm_cvsearch.%N.%A_%a.log \
#          --export=ALL,PY_SCRIPT=src/training/train_mimic_resnet18.py,SWAD=1 \
#          slurm/c2_mimic_cv_search.sh
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
# semois GPU 反复 device-handle 故障，永久排除（C 阶段纪律）
#SBATCH --exclude=semois
#SBATCH --array=0-29

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

CV=5                                       # 5 折患者级 GroupKFold（cv5/fold{0..4} 已就位）
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID % 6))  # 0–5：超参网格 config
FOLD=$((SLURM_ARRAY_TASK_ID / 6))          # 0–4：折号
SEED=$((42 + FOLD))                        # fold↔seed 一一映射，避免 run_id 覆盖

echo "C2_MIMIC_CV_SEARCH  PY_SCRIPT=$PY_SCRIPT  config_index=$CONFIG_INDEX  fold=$FOLD  seed=$SEED  cv=$CV  BATCH=${BATCH:-config}  SWAD=${SWAD:-0}"

python "$PY_SCRIPT" \
    --config_index "$CONFIG_INDEX" \
    --seed "$SEED" \
    --cv "$CV" \
    --fold "$FOLD" \
    ${BATCH:+--batch_size "$BATCH"} \
    ${SWAD:+--swad}
