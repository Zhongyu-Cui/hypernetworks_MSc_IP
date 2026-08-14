#!/bin/bash
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00
#SBATCH --job-name=e2_mimic_deep_redo
#SBATCH --exclude=semois,mira05,deepmedic2
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/e2_mimic_deep_redo.%N.%j.log
source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1
python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_condnet.py \
  --dataset mimic --location deep --config_index 3 --seed 42 --cv 5 --fold 2
