#!/bin/bash
#SBATCH --job-name=crosspoint_newFSData_simclr_noNorm_test
#SBATCH --output=logs/crosspoint_newFSData_simclr_noNorm_test.txt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=a100_80|h100
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00

ml anaconda/latest
ml cuda/12.1.1_530.30.02
ml cudnn/8.9.7.29_cuda12

conda activate crosspoint_2dtm

cd /home/kithmini.herath/codes/CrossPoint
# 250 e
python train_crosspoint_fsdata.py --model dgcnn --dataset new_data/ds_original --epochs 2 --exp_name crosspoint_newFSData_simclr_noNorm_noise2d_0_test --batch_size 64 --print_freq 20 --k 15 --save_freq 10 --noise2d 0.0 --loss SimCLR --test True