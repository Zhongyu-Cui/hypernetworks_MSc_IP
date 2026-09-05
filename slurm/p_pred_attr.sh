#!/bin/bash
# ============================================================
# 实验 P·Predicted-Attribute HyperAdapt（g 的 softmax 替代真值属性）
# ============================================================
# 方案：docs/predicted_attribute_hyperadapt_plan.md
# g（冻结 ImageNet 特征 + logistic probe）已由 scripts/build_attr_predictions.py 逐 fold
# 在 train 上拟合并出好 p̂（in-sample，无交叉拟合），本作业只做第二阶段训练。
#
# array 0–4 → (fold, seed)：FOLD = task_id，SEED = 42 + FOLD（与 GT 臂 cv5 的 fold↔seed 映射一致，
# run_id 不含 fold，靠 seed 区分，产物与 GT 臂同址 outputs/<ds>/cv5/ 但 method 名不同，互不覆盖）。
#
# 经 --export 注入：
#   PY_SCRIPT    : 训练脚本绝对路径（如 .../train_fitzpatrick_hyperadapt_pred.py）
#   ATTR_MODE    : soft(主线) / hard(消融) / perm(路由对照) / const(容量对照)
#   CONFIG_INDEX : 超参网格序号；缺省 5 = lr3e-04_wd1e-03 = Fitzpatrick GT-HyperAdapt 选定配置
#                  （HAM=0 / MIMIC=3 / CheXpert=4 / PAPILA=5，均为各库 GT 臂的 Pareto 选定值）
#   DS_ARG       : 仅通用 CXR 脚本需要，mimic / chexpert
#   COND         : 可选，仅 HAM pred 脚本支持——条件通路 age（缺省，历史臂）/ sex_age（双属性对照臂，
#                  method 记为 <method>_sexage，p̂ 取自 attr_pred_sexage/）
#   TRIAL        : 可选，σ_train replicate 的 trial 号（0=主结果，1/2=额外重复；见下方 SEED 计算）
#   BATCH        : 可选，覆盖 batch_size（须与 GT 臂等 batch；Fitzpatrick config 已是 128）
#
# 示例（Fitzpatrick 主线 soft 全 5 折）：
#   sbatch --partition=gpus48 --job-name=p_fitz_soft \
#          --output=/vol/.../logs/p_fitz_soft.%N.%A_%a.log \
#          --export=ALL,PY_SCRIPT=/vol/.../src/training/train_fitzpatrick_hyperadapt_pred.py,ATTR_MODE=soft \
#          slurm/p_pred_attr.sh
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00:00
# semois GPU 反复 device-handle 故障，永久排除（C 阶段纪律）
#SBATCH --exclude=semois
#SBATCH --array=0-4

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

if [[ -z "$PY_SCRIPT" ]]; then
    echo "ERROR: 必须经 --export 传入 PY_SCRIPT=<训练脚本绝对路径>" >&2
    exit 1
fi
if [[ -z "$ATTR_MODE" ]]; then
    echo "ERROR: 必须经 --export 传入 ATTR_MODE=<soft|hard|perm|const>" >&2
    exit 1
fi

CV=5
FOLD=$SLURM_ARRAY_TASK_ID
# TRIAL>0 = σ_train replicate：同 fold、同 config、只换训练随机种子。
# seed = 42 + fold + 100*trial ⇒ trial0=42..46（主结果）/ trial1=142..146 / trial2=242..246，
# run_id 天然不冲突，既有产物零覆盖风险。
TRIAL=${TRIAL:-0}
SEED=$((42 + FOLD + 100 * TRIAL))
CONFIG_INDEX=${CONFIG_INDEX:-5}

echo "=== 实验 P: $(basename "$PY_SCRIPT")  mode=$ATTR_MODE  cond=${COND:-age}  fold=$FOLD  seed=$SEED  config=$CONFIG_INDEX ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CMD=(python "$PY_SCRIPT" --attr_mode "$ATTR_MODE" --config_index "$CONFIG_INDEX"
     --cv "$CV" --fold "$FOLD" --seed "$SEED")
# 通用 CXR 脚本（train_cxr_hyperadapt_pred.py）用 --dataset 区分 mimic / chexpert
if [[ -n "$DS_ARG" ]]; then
    CMD+=(--dataset "$DS_ARG")
fi
if [[ -n "$BATCH" ]]; then
    CMD+=(--batch_size "$BATCH")
fi
# HAM 双属性对照臂：条件输入补齐为 sex+age（缺省 age，历史行为不变）
if [[ -n "$COND" ]]; then
    CMD+=(--cond "$COND")
fi
echo "CMD: ${CMD[*]}"
"${CMD[@]}"
