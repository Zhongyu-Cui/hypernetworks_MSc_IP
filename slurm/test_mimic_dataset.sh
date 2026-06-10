#!/bin/bash
#SBATCH --partition=gpus                                  # 轻量自检任务，无需独占高端 GPU
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/test_mimic_dataset.%N.%j.log
#SBATCH --time=0-00:30:00
#SBATCH --job-name=test_mimic_dataset

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/datasets/mimic_cxr_dataset.py
