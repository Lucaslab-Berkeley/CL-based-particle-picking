#!/bin/bash
#SBATCH --job-name=crosspoint_dgcnn_cls
#SBATCH --output=crosspoint_dgcnn_cls.txt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=a100|h100
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00

ml anaconda/latest
ml cuda/12.1.1_530.30.02
ml cudnn/8.9.7.29_cuda12

conda activate crosspoint_2dtm

cd /home/kithmini.herath/codes/CrossPoint

# Traning CrossPoint for classification, epochs = 100
python train_crosspoint_Original.py --model dgcnn --epochs 100 --lr 0.001 --exp_name crosspoint_dgcnn_cls --batch_size 20 --print_freq 20 --k 15 --save_freq 10
