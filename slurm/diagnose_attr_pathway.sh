#!/bin/bash
#SBATCH --partition=gpus                                            # 纯推理任务，用低优先级混杂分区即可
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/diagnose_attr_pathway.%N.%j.log
#SBATCH --time=0-02:00:00                                           # 5 个模型 × 测试集多次前向，2h 足够
#SBATCH --job-name=diagnose_attr_pathway

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
export PYTHONUNBUFFERED=1

python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/scripts/diagnose_attr_pathway.py
