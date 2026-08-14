#!/bin/bash
# ============================================================
# OOD v2 · 阶段 3：full-target 终评（双向 × 3 方法 × 5 fold × 3 trial）
# ============================================================
# 预注册：docs/ood_experiment_v2_preregistration.md §3（主估计量 = full-target）。
# 每个 source (fold, trial) 模型评**完整** target，逐 (f,t) 算指标再等权平均 —— 从不跨折池化，
# 免疫跨折校准漂移，且逐 (f,t) 天然配对。
#
# array 0–5 → (方向, 方法)：
#   DIRECTION = task < 3 ? m2c : c2m
#   METHOD    = (erm, swad, hyperadapt)[task % 3]
# trial 全传 0,1,2；已存在的预测自动跳过（M→C trial 0 已由 E-audit.2 产出，共 25 个）。
#
# 用法：
#   sbatch --partition=gpus24 --job-name=oodv2_ft \
#          --output=/vol/biomedic2/bglocker_studproj/zc125/logs/oodv2_ft.%N.%A_%a.log \
#          slurm/oodv2_full_target.sh
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
# semois GPU 反复 device-handle 故障；mira05 间歇性 "GPU is lost / unspecified launch failure"
# （作业 73768_5 在此节点跑完 4/15 后崩溃；同节点训练却正常 ⇒ 长时推理下才暴露）。两者永久排除。
#SBATCH --exclude=semois,mira05
#SBATCH --array=0-5

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1
cd /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP

METHODS=(erm swad hyperadapt)
if [[ -n "$METHOD_OVERRIDE" ]]; then
    # 实验 G（HN×SWAD 融合臂）追加用法：单方法双向，提交时配 --array=0-1
    #   task 0 → m2c，task 1 → c2m
    # METHOD_OVERRIDE 未设时下方原逻辑逐位不变 ⇒ v2 预注册的既有提交方式不受影响。
    METHOD=$METHOD_OVERRIDE
    if (( SLURM_ARRAY_TASK_ID == 0 )); then DIRECTION=m2c; else DIRECTION=c2m; fi
else
    if (( SLURM_ARRAY_TASK_ID < 3 )); then DIRECTION=m2c; else DIRECTION=c2m; fi
    METHOD=${METHODS[$((SLURM_ARRAY_TASK_ID % 3))]}
fi

echo "OODV2_FULL_TARGET  direction=$DIRECTION  method=$METHOD  trials=0,1,2"
# 全 target 推理是 I/O 密集（MIMIC 19.9 万张走 NFS）；worker=4 在高负载节点上会让 GPU 饿死
# （作业 73789 实测 GPU 0%、单模型 >31 min 未完成）。gpus24 分区每节点 12 CPU，取 8 留余量。
python -m src.training.run_ood_cxr_full_target \
    --direction "$DIRECTION" \
    --method "$METHOD" \
    --trials 0,1,2 \
    --num-workers "${NUM_WORKERS:-8}"
echo "oodv2_full_target done rc=$?"
