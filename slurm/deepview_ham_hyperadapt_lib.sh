#!/bin/bash
#SBATCH --partition=gpus24                                              # inference only; HyperAdapt per-sample conv is memory-heavy but fits gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/deepview_ham_hyperadapt_lib.%N.%j.log
#SBATCH --time=0-01:30:00                                               # official DeepView Fisher distance is numpy/per-row => O(N^2), slow; leave headroom
#SBATCH --job-name=deepview_ham_lib

# Official-DeepView decision-boundary viz (HAM10000 age-conditioned HyperAdapt, 4 age groups).
# Inference only. The DeepView library's Fisher distance is a numpy per-row loop, much slower than
# a vectorised GPU version, so prefer the cluster for n>=100 / resolution>=100.
# Optional env vars: CKPT / N / RES / ANCHOR / SEED.

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

CKPT=${CKPT:-/vol/biomedic2/bglocker_studproj/zc125/outputs/ham10000/hyperadapt_lr3e-04_wd1e-04_seed42_best_overall.pth}
N=${N:-120}
RES=${RES:-100}
ANCHOR=${ANCHOR:-1}
SEED=${SEED:-42}

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/scripts/deepview_ham_hyperadapt_lib.py \
    --ckpt "$CKPT" --n_samples "$N" --resolution "$RES" --anchor_age "$ANCHOR" --seed "$SEED"
