#!/bin/bash
#SBATCH --partition=gpus24                                              # inference only; ERM image-only is light
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/deepview_mimic_erm.%N.%j.log
#SBATCH --time=0-01:00:00                                               # two datasets x DeepView numpy Fisher; ERM forward cheap
#SBATCH --job-name=deepview_erm

# Official-DeepView for the ERM image-only baseline: MIMIC (ID) vs CheXpert (OOD), two panels.
# ERM has no attribute conditioning -> one boundary per dataset. tau=0.305 (MIMIC operating point).
# Optional env vars: CKPT / N / RES / SEED.

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

CKPT=${CKPT:-/vol/biomedic2/bglocker_studproj/zc125/outputs/mimic_cxr/erm_lr3e-04_wd1e-03_seed42_best_overall.pth}
N=${N:-110}
RES=${RES:-90}
SEED=${SEED:-42}

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/scripts/deepview_mimic_erm_lib.py \
    --ckpt "$CKPT" --n_samples "$N" --resolution "$RES" --seed "$SEED"
