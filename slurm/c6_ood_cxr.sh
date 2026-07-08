#!/bin/bash
# ============================================================
# C6 OOD 全量评估（CXR：MIMIC↔CheXpert 双向）—— 单作业推理，无训练
# ============================================================
# 加载 MIMIC/CheXpert 确认配置 checkpoint，在对端 test 集评估 5 方法 × 5 seed × 2 方向，
# 落盘 OOD 预测 npz（outputs/ood_cxr/<src>2<tgt>/）。纯推理（~50 次 forward，目标 test 12–17k 图），
# 走 gpus 分区即可。方向由 --export DIRECTION 注入（默认 both）。
#
# 示例：
#   sbatch --partition=gpus --job-name=c6_ood --output=/vol/.../logs/c6_ood.%N.%j.log \
#          --export=ALL,DIRECTION=both slurm/c6_ood_cxr.sh
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --exclude=semois

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

echo "C6_OOD  DIRECTION=${DIRECTION:-both}"
python -m src.training.run_ood_cxr_all --direction "${DIRECTION:-both}"
