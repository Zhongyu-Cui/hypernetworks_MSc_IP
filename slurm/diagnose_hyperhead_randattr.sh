#!/bin/bash
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/diagnose_hyperhead_randattr.%N.%A_%a.log
#SBATCH --time=1-12:00:00
#SBATCH --job-name=diag_randattr
#SBATCH --array=0-1

# 干净正控制（修正 XOR-sex 被胸片性别泄漏污染的问题）：
# 注入与图像无关的随机二值属性 r，标签 = label XOR r，r 经 sex 槽位喂给模型。
#   0  HyperHead   : 拿到 r → 应恢复 AUC ≈baseline (~0.85)
#   1  image-only  : 看不到 r → 在 y XOR r 上应坍到 AUC ≈0.5（下界锚点）
# 判据：job0 ≈0.85 且 job1 ≈0.5 ⇒ 超网络机器健康、能利用属性，工程问题排除。

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

SCRIPT=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/diagnose_hyperhead_pathway.py
SEED=42

case "$SLURM_ARRAY_TASK_ID" in
  0) ARGS="--model hyperhead  --random_attr --tag pc_hyper_randattr" ;;
  1) ARGS="--model imageonly  --random_attr --tag pc_imageonly_randattr" ;;
  *) echo "Unknown SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

echo "Running task $SLURM_ARRAY_TASK_ID: $ARGS"
python "$SCRIPT" --seed "$SEED" $ARGS
