#!/bin/bash
# ============================================================
# HAM10000 全-CV 超参搜索（6 config × 5 折）——选择信号有效性验证前提
# ============================================================
# 动机：selection_strategy_probe 证「单-split 下 val 选择信号与 test worst-group 反相关
# （ρ≈−0.2~−0.3，全策略失效）」。要检验「fold-mean 折均能否把这个负相关救成正」，须有
# **6 config × 5 折**的 CV 预测（现有路线 A 只训了选定 1 config × 5 折，无法做折均选择）。
# 本脚本对齐 PAPILA 的 c_search_papila_cv.sh，把 HAM 从「单-split + 5-seed」升级为「6 config × 5 折」，
# 使 config 选择走 CV 内折均-minimax（PAPILA 式，无泄漏），并让 OOF 池化(n=9707)成为可信 test 目标。
#
# 与路线 A(c1_ham_cv_oof.sh)的区别：路线 A array 0–4（固定 config × 5 折）；本脚本 array 0–29
# （6 config × 5 折全跑），是 Q3「CV 内重搜超参」的落地，产出折均 Pareto/DTO 所需的候选池。
#
# array 0–29 → (config_index, fold, seed) 三元解码（同 PAPILA 约定）：
#   CONFIG_INDEX = task % 6      （6 配置 lr∈{3e-5,1e-4,3e-4}×wd∈{1e-4,1e-3}）
#   FOLD         = task / 6      （5 折 lesion 级 GroupKFold，cv5/fold{0..4} 已就位）
#   SEED         = 42 + FOLD     （fold↔seed 一一映射：run_id 不含 fold，5 折共用 seed 会覆盖预测）
# 每方法一次提交产出 6 config × 5 fold = 30 条 val 日志 + OOF test 预测，供折均选择与代理秩相关验证。
#
# 数据集/方法差异由 sbatch 命令行 + --export 注入：
#   --partition=<gpus24|gpus48>   HyperAdapt bs128 用 gpus48 等-batch，余 gpus24
#   --job-name / --output         作业名 / 日志路径
#   --export=ALL,PY_SCRIPT=<abs.py>[,BATCH=<n>][,SWAD=1]
#     PY_SCRIPT : HAM 训练脚本绝对路径（train_ham10000_*.py，均支持 --cv/--fold/--config_index）
#     BATCH     : 可选，覆盖 config 的 batch_size（HyperAdapt 等-batch 传 128）
#     SWAD      : 可选，=1 时加 --swad（仅 ERM 用，逐折派生 SWAD 基线）
#     COND      : 可选，=sex_age 时加 --cond sex_age（**双属性对照臂**：三个 HAM HN 脚本把条件
#                 输入由 age-only 补齐为 sex+age，method 记为 <method>_sexage，与 age-only 臂
#                 产物分开存放。动机：worst-group 评估口径是 Sex/Age/Sex×Age，而 age-only 臂
#                 从未拿到 sex，条件化与评估口径不对称。缺省=age，历史行为不变）
#
# 示例（HAM ERM 全-CV 搜索 + 逐折 SWAD 派生）：
#   sbatch --partition=gpus24 --job-name=ham_erm_cvsearch \
#          --output=/vol/biomedic2/bglocker_studproj/zc125/logs/ham_erm_cvsearch.%N.%A_%a.log \
#          --export=ALL,PY_SCRIPT=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_ham10000_resnet18.py,SWAD=1 \
#          slurm/c1_ham_cv_search.sh
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

CV=5                                       # 5 折 lesion 级 GroupKFold（cv5/fold{0..4} 已就位）
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID % 6))  # 0–5：超参网格 config
FOLD=$((SLURM_ARRAY_TASK_ID / 6))          # 0–4：折号
SEED=$((42 + FOLD))                        # fold↔seed 一一映射，避免 run_id 覆盖

echo "C1_HAM_CV_SEARCH  PY_SCRIPT=$PY_SCRIPT  config_index=$CONFIG_INDEX  fold=$FOLD  seed=$SEED  cv=$CV  BATCH=${BATCH:-config}  SWAD=${SWAD:-0}  COND=${COND:-age}"

python "$PY_SCRIPT" \
    --config_index "$CONFIG_INDEX" \
    --seed "$SEED" \
    --cv "$CV" \
    --fold "$FOLD" \
    ${BATCH:+--batch_size "$BATCH"} \
    ${SWAD:+--swad} \
    ${COND:+--cond "$COND"}
