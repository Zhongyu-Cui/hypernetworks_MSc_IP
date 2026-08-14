#!/bin/bash
# ============================================================
# OOD 属性 knockout 反事实（机制 A vs 机制 B）
# ============================================================
# 见 src/training/run_ood_attr_knockout.py 与 docs/ood_experiment_v2_results.md §5.1。
# 同一份 HyperAdapt 权重，把喂给超网络的属性换成 permute / constant 再在完整 target 上推断；
# 与既有的 real（cv5_full_target）对比 ⇒ 判断 OOD 增益是否依赖属性。
#
# array 0–2 → mode：0=permute（保边缘分布，破配对）；1=constant（全置 majority cell）；
#              2=marginal（输出边缘化，8 组合按 target 先验加权平均概率——严格版）
# 已存在的预测自动跳过，故重投时 permute/constant 不会重算。
# 每个 mode 跑 5 fold × 3 trial = 15 次完整 target 推理。
#
# 用法：
#   sbatch --partition=gpus24 --job-name=ood_knockout \
#          --output=/vol/biomedic2/bglocker_studproj/zc125/logs/ood_knockout.%N.%A_%a.log \
#          slurm/ood_knockout.sh
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
# semois device-handle 故障；mira05 间歇性 "GPU is lost"（作业 73768_5）；
# deepmedic2 长期 CPU 超载（实测 load 16/12 核，GPU 恒 0% 被 DataLoader 饿死；作业 73789、73898）。
#SBATCH --exclude=semois,mira05,deepmedic2
# %1 = 串行执行：两个 mode 各开 8 个 DataLoader worker，若并发落到同一节点会自相争用 12 核。
#SBATCH --array=0-2%1

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1
cd /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP

MODES=(permute constant marginal)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

echo "OOD_KNOCKOUT  direction=${DIRECTION:-m2c}  mode=$MODE  trials=0,1,2"
python -m src.training.run_ood_attr_knockout \
    --direction "${DIRECTION:-m2c}" \
    --mode "$MODE" \
    --trials 0,1,2 \
    --num-workers "${NUM_WORKERS:-8}"
echo "ood_knockout done rc=$?"
