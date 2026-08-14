#!/bin/bash
# ============================================================
# 超参匹配 ERM 对照的 OOD 推断（M→C / C→M）—— GPU 推理作业
# ============================================================
# 背景：CV-OOF OOD 里 HyperAdapt 显著胜 ERM（M→C +0.0045 / C→M +0.0017），但两者**选中的超参不同**：
#   M→C：ERM=lr1e-04_wd1e-04  vs  HyperAdapt=lr1e-04_wd1e-03（wd 大 10×）
#   C→M：ERM=lr1e-04_wd1e-04  vs  HyperAdapt=lr3e-04_wd1e-04（lr 大 3×）
# 两个方向 HyperAdapt 都落在更强正则/更大 lr 一侧 ⇒ 「HN 赢」可能只是 Minimax-Pareto 选择器替 HN
# 选到了更 OOD-友好的超参。本作业用**与 HyperAdapt 同超参的 ERM**（由 hpmatch_{mimic,chex}_erm
# 训练作业产出）跑同一套 OOD 评估，破除该混杂。
#
# 输出写 outputs/ood_cxr/{s}2{t}/cv5_hpmatch/predictions/，**与主结果 cv5/ 完全隔离**。
#
# 用法：
#   sbatch --partition=gpus24 --job-name=hpmatch_ood \
#          --output=/vol/biomedic2/bglocker_studproj/zc125/logs/hpmatch_ood.%N.%j.log \
#          slurm/hpmatch_ood_cv.sh
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
# semois GPU 反复 device-handle 故障，永久排除（C 阶段纪律）
#SBATCH --exclude=semois

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1
cd /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP

echo "HPMATCH_OOD  只评 ERM，超参对齐各方向的 HyperAdapt 选中配置"
python -m src.training.run_ood_cxr_cv \
    --direction both \
    --methods erm \
    --config-m2c erm=lr1e-04_wd1e-03 \
    --config-c2m erm=lr3e-04_wd1e-04 \
    --out-subdir cv5_hpmatch
echo "hpmatch_ood done rc=$?"
