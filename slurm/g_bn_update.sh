#!/bin/bash
# ============================================================
# 实验 G · SWAD 的 BN 统计与官方 SWAD 对齐（update_bn）+ 重推理
# ============================================================
# 官方 swad（khanrc/swad, domainbed/trainer.py）在权重平均后调用
#   swa_utils.update_bn(train_minibatches_iterator, swad_algorithm, n_steps=500)
# 重估 BN running stats；本项目原实现是对 running_mean/var 直接算术平均。本作业补齐该步骤，
# 检验 docs/hyperadapt_swad_fusion_full_report.md §5.3.4② 的「SWAD 压低 calibration slope」
# 是否源于 BN 统计失配。**不重训**，仅无梯度前向 + 重推理。
#
# 产物全部隔离，既有冻结产物不被覆盖：
#   checkpoint  outputs/<ds>/cv5/*_averaged_bnupd.pth
#   ID 预测     outputs/<ds>/cv5_bnupd/predictions/
#   OOD 预测    outputs/ood_cxr/<dir>/cv5_full_target_bnupd/predictions/
#
# 经 --export 注入：
#   DATASET : ham10000 | fitzpatrick | mimic | chexpert   （必填）
#   STAGE   : id | ood | bn-only                          （必填）
#   TRIALS  : trial 编号，缺省 0（OOD 用 0:1:2）
#   FOLDS   : 缺省 0:1:2:3:4
#
# ⚠️ **多值一律用冒号分隔，不能用逗号**：sbatch `--export` 本身以逗号切分变量，
#    `--export=ALL,STAGE='id,ood'` 会被解析成 `STAGE=id` 加一个名为 `ood` 的空变量
#    （作业 75315/75316 首次提交即因此只跑了 id 阶段）。脚本内部再把冒号转回去。
#
# 示例：
#   sbatch --partition=gpus24 --job-name=g_bn_ham \
#          --output=logs/g_bn_ham.%N.%j.log \
#          --export=ALL,DATASET=ham10000,STAGE=id slurm/g_bn_update.sh
#   sbatch --partition=gpus48 --job-name=g_bn_mimic_ood \
#          --output=logs/g_bn_mimic_ood.%N.%j.log \
#          --export=ALL,DATASET=mimic,STAGE=ood,TRIALS=0:1:2 slurm/g_bn_update.sh
#
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00:00
# semois：GPU device-handle 故障，永久排除（C 阶段纪律）
# mira05：长时推理下 GPU lost（OOD v2 记录）
#SBATCH --exclude=semois,mira05

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

if [[ -z "$DATASET" || -z "$STAGE" ]]; then
    echo "ERROR: 必须经 --export 传入 DATASET 与 STAGE" >&2
    exit 1
fi

# 冒号 → 脚本所需形式（见顶部 --export 逗号陷阱说明）
STAGE_LIST="${STAGE//:/ }"
TRIALS_ARG="${TRIALS:-0}"; TRIALS_ARG="${TRIALS_ARG//:/,}"
FOLDS_ARG="${FOLDS:-0:1:2:3:4}"; FOLDS_ARG="${FOLDS_ARG//:/,}"

echo "G_BN_UPDATE  DATASET=$DATASET  STAGE=$STAGE_LIST  TRIALS=$TRIALS_ARG  FOLDS=$FOLDS_ARG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# STAGE 支持多值（如 id:ood）：同一作业内串行，避免两个作业并发写同一个 bnupd checkpoint。
# 第二个 stage 的 BN 重估会因文件已存在而全部跳过，只做推理。
rc=0
for S in $STAGE_LIST; do
    echo "--- stage=$S ---"
    python "$REPO_ROOT/scripts/g_bn_update_swad.py" \
        --dataset "$DATASET" \
        --stage "$S" \
        --trials "$TRIALS_ARG" \
        --folds "$FOLDS_ARG" || rc=$?
done

echo "g_bn_update done rc=$rc"
