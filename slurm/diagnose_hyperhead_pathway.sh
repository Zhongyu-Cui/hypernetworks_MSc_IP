#!/bin/bash
#SBATCH --partition=gpus24                                                      # 单卡，结构同 baseline，24GiB 足够
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/diagnose_hyperhead_pathway.%N.%A_%a.log
#SBATCH --time=1-12:00:00
#SBATCH --job-name=diag_hyperhead
#SBATCH --array=0-3                                                             # 4 个诊断配置（见下方 CASE）

# 诊断「超网络不提升公平性是工程问题还是信号弱」的四个 job（均 seed 42，先做一轮）：
#   0  正控制·HyperHead   : 标签=label XOR sex，期望 AUC 恢复到 ≈baseline ⇒ 机器健康
#   1  正控制·image-only  : 标签=label XOR sex，看不到属性，期望 AUC≈0.5（反向对照下界）
#   2  尺度扫描·放大通路   : 真实标签，hyper lr×10 + init_std 0.05（同容量）
#   3  尺度扫描·放大+扩容  : 真实标签，hyper lr×10 + init_std 0.05 + hidden256 + embed16
# 判据：job0≫job1 ⇒ 工程实现没问题；job2/3 的 worst-case AUC 是否超过 baseline 的 0.8171。

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

SCRIPT=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/diagnose_hyperhead_pathway.py
SEED=42

case "$SLURM_ARRAY_TASK_ID" in
  0)
    ARGS="--model hyperhead --synthetic_xor sex --tag pc_hyper_xorsex"
    ;;
  1)
    ARGS="--model imageonly --synthetic_xor sex --tag pc_imageonly_xorsex"
    ;;
  2)
    ARGS="--model hyperhead --hyper_lr_mult 10 --hyper_init_std 0.05 --tag scale_amp"
    ;;
  3)
    ARGS="--model hyperhead --hyper_lr_mult 10 --hyper_init_std 0.05 --hyper_hidden 256 --hyper_embed 16 --tag scale_ampcap"
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
    exit 1
    ;;
esac

echo "Running task $SLURM_ARRAY_TASK_ID: $ARGS"
python "$SCRIPT" --seed "$SEED" $ARGS
