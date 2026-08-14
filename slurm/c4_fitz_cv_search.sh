#!/bin/bash
# ============================================================
# Fitzpatrick17k 全-CV 超参搜索（6 config × 5 折）——CV-OOF 口径铺开（方案 R-Fitz）
# ============================================================
# 对齐 slurm/c1_ham_cv_search.sh：把 Fitzpatrick 从「单-split」升级为「6 config × 5 折」CV 搜索，
# 使 config 选择走折均 marginal-worst（scripts/select_config_cv.py），评估走 pooled-OOF（n=16012，
# scripts/cv_oof_report.py）。Fitzpatrick 敏感属性=肤色 skin（单一 6 类轴），无 patient 分组
# （每 md5hash 唯一），故 CV 折用**图像级 StratifiedKFold**（cv5/fold{0..4} 由
# scripts/build_fitzpatrick_cv_splits.py 生成，已断言并集=16012、5 折 test disjoint）。
#
# array 0–29 → (config_index, fold, seed) 三元解码（同 HAM/PAPILA 约定）：
#   CONFIG_INDEX = task % 6      （6 配置 lr∈{3e-5,1e-4,3e-4}×wd∈{1e-4,1e-3}）
#   FOLD         = task / 6      （5 折图像级 StratifiedKFold，cv5/fold{0..4} 已就位）
#   SEED         = 42 + FOLD     （fold↔seed 一一映射：run_id 不含 fold，5 折共用 seed 会覆盖预测）
# 每方法一次提交产出 6 config × 5 fold = 30 条 val 日志 + OOF test 预测，写 outputs/fitzpatrick/cv5/。
#
# 数据集/方法差异由 sbatch 命令行 + --export 注入：
#   --partition=<gpus24|gpus48>   HyperAdapt 等-batch bs128 用 gpus48（避免 HAM 的 b32 OOM 伪影教训），余 gpus24
#   --job-name / --output         作业名 / 日志路径
#   --export=ALL,PY_SCRIPT=<abs.py>[,BATCH=<n>][,SWAD=1]
#     PY_SCRIPT : Fitzpatrick 训练脚本绝对路径（train_fitzpatrick_*.py，均支持 --cv/--fold/--config_index）
#     BATCH     : 可选，覆盖 config 的 batch_size（HyperAdapt 等-batch 传 128）
#     SWAD      : 可选，=1 时加 --swad（仅 ERM 用，逐折派生 SWAD 基线）
#
# 示例（Fitzpatrick ERM 全-CV 搜索 + 逐折 SWAD 派生）：
#   sbatch --partition=gpus24 --job-name=fitz_erm_cvsearch \
#          --output=/vol/biomedic2/bglocker_studproj/zc125/logs/fitz_erm_cvsearch.%N.%A_%a.log \
#          --export=ALL,PY_SCRIPT=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_fitzpatrick_resnet18.py,SWAD=1 \
#          slurm/c4_fitz_cv_search.sh
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
# semois GPU 反复 device-handle 故障，永久排除（C 阶段纪律）
#SBATCH --exclude=semois
#SBATCH --array=0-29

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

if [[ -z "$PY_SCRIPT" ]]; then
    echo "ERROR: 必须经 --export 传入 PY_SCRIPT=<训练脚本绝对路径>" >&2
    exit 1
fi

CV=5                                       # 5 折图像级 StratifiedKFold（cv5/fold{0..4} 已就位）
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID % 6))  # 0–5：超参网格 config
FOLD=$((SLURM_ARRAY_TASK_ID / 6))          # 0–4：折号
SEED=$((42 + FOLD))                        # fold↔seed 一一映射，避免 run_id 覆盖

echo "C4_FITZ_CV_SEARCH  PY_SCRIPT=$PY_SCRIPT  config_index=$CONFIG_INDEX  fold=$FOLD  seed=$SEED  cv=$CV  BATCH=${BATCH:-config}  SWAD=${SWAD:-0}"

python "$PY_SCRIPT" \
    --config_index "$CONFIG_INDEX" \
    --seed "$SEED" \
    --cv "$CV" \
    --fold "$FOLD" \
    ${BATCH:+--batch_size "$BATCH"} \
    ${SWAD:+--swad}
