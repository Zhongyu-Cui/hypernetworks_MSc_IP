#!/bin/bash
#SBATCH --partition=gpus24                                              # inference only; HyperAdapt per-sample conv is memory-heavy but fits gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/deepview_mimic_hyperadapt_lib.%N.%j.log
#SBATCH --time=0-01:30:00                                               # DeepView Fisher distance is numpy/per-row O(N^2), slow; 8 subgroups x grid re-scoring
#SBATCH --job-name=deepview_mimic_lib

# Official-DeepView decision-boundary viz (MIMIC-CXR HyperAdapt, 8 sex/race/age subgroups).
# Inference only. Optional env vars: CKPT / N / RES / ANCHOR / SEED.

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

CKPT=${CKPT:-/vol/biomedic2/bglocker_studproj/zc125/outputs/mimic_cxr/hyperadapt_lr3e-04_wd1e-04_seed42_best_overall.pth}
N=${N:-120}
RES=${RES:-90}
ANCHOR=${ANCHOR:-1}
SEED=${SEED:-42}

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/scripts/deepview_mimic_hyperadapt_lib.py \
    --ckpt "$CKPT" --n_samples "$N" --resolution "$RES" --anchor_idx "$ANCHOR" --seed "$SEED"
