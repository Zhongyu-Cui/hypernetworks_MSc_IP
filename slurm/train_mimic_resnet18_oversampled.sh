#!/bin/bash
# DEPRECATED: 过采样策略已废弃，使用 slurm/train_mimic_resnet18.sh 提交 No Finding 训练。
#SBATCH --partition=gpus24                                          # 单卡训练首选分区 (24GiB 足够 ResNet-18 @ 224x224, batch_size=64)
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic2/bglocker_studproj/zc125/logs/train_mimic_resnet18_oversampled.%N.%j.log
#SBATCH --time=1-12:00:00                                           # 早停上限 20 epoch；过采样后训练集增大约 1.53x (213,862 -> 327,280)，留出余量覆盖单 epoch 用时
#SBATCH --job-name=train_mimic_resnet18_oversampled

# 与 train_mimic_resnet18.sh 跑的是同一个训练脚本/同一套早停与超参设置，
# 区别只在于数据预处理流水线已切换为 2026-06-08 更新版本：
#   1) 训练集改用 train_oversampled.csv (正例占比 23.5% -> 50%，缓解类别不平衡)
#   2) 训练 transform 关闭 RandomHorizontalFlip (CXR 有固定解剖学左右关系)
# 详见仓库 CLAUDE.md「训练集过采样」决策记录。
# checkpoint 写出到 resnet18_pleural_effusion_oversampled_best.pth，
# 不会覆盖旧 baseline 的 resnet18_pleural_effusion_best.pth，便于横向对比。

source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
python /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_mimic_resnet18.py
