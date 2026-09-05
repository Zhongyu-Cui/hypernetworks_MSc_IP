#!/bin/bash
# ============================================================
# CV-OOF pooled 显著性报告（cv_oof_report.py --section sig）作为 SLURM 作业
# ============================================================
# 动机：大数据集（CheXpert n≈138k、MIMIC n≈199k）的样本级配对 bootstrap（18 组比较 × 2000 次 ×
# 14 子群 AUC）在实验室机器上过重（违反「实验室机器仅轻量任务」纪律），改在 SLURM 节点跑。
# 纯 CPU（numpy/sklearn，无 GPU），但集群按 GPU 分配，故 --gres=gpu:1（不使用 GPU）。
#
# 用法（经 --export 注入 DATASET；结果落 logs 与 --out）：
#   sbatch --partition=gpus --job-name=mimic_sig \
#          --output=logs/mimic_sig.%N.%j.log \
#          --export=ALL,DATASET=mimic,CONFIG_JSON=$HN_WORK_ROOT/outputs/mimic_cxr/cv5/selected_configs.json,OUT=$HN_WORK_ROOT/scratch/mimic_sig.txt \
#          slurm/cv_oof_sig.sh
#SBATCH --gres=gpu:1
#SBATCH --time=0-03:00:00
#SBATCH --exclude=semois

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
cd "$REPO_ROOT"

if [[ -z "$DATASET" || -z "$OUT" ]]; then
    echo "ERROR: 需 --export DATASET / OUT（配置二选一：CONFIG_JSON 或 CONFIG）" >&2
    exit 1
fi
# 配置来源二选一：CONFIG_JSON（ID 用选定 JSON）或 CONFIG（OOD 用 method=tag 空格串 + 可选 PRED_DIR）
if [[ -n "$CONFIG_JSON" ]]; then
    CFG_ARG="--config-json $CONFIG_JSON"
else
    CFG_ARG="--config $CONFIG"
fi

echo "CV_OOF_SIG  dataset=$DATASET  pred_dir=${PRED_DIR:-default}  out=$OUT  n_boot=${NBOOT:-2000}"
python scripts/cv_oof_report.py --dataset "$DATASET" $CFG_ARG --section sig \
    ${PRED_DIR:+--pred-dir "$PRED_DIR"} ${NBOOT:+--n-boot "$NBOOT"} > "$OUT" 2>&1
echo "sig done rc=$? -> $OUT"
