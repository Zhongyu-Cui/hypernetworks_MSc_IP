#!/bin/bash
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/probe_condnet.%N.%j.log
#SBATCH --time=0-00:20:00
#SBATCH --job-name=probe_condnet
#SBATCH --exclude=semois,mira05
source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1
python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/scripts/probe_condnet_memory.py --batch_size 128
