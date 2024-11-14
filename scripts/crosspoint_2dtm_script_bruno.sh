#!/bin/bash
#SBATCH --job-name=crosspoint_2dtm_2_onTheFly_2classes_simclr
#SBATCH --output=logs/crosspoint_2dtm_2_onTheFly_2classes_simclr.txt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=a100_80|h100
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00

ml anaconda/latest
ml cuda/12.1.1_530.30.02
ml cudnn/8.9.7.29_cuda12

conda activate crosspoint_2dtm

cd /home/kithmini.herath/codes/CL-based-particle-picking

# to set the constraint for slurm: --constraint=a100_80
# crosspoint_2dtm_2 : val_included 0
# crosspoint_2dtm_4 : val_inlcuded 1
# Traning CrossPoint for classification, epochs = 100, --val_included 1
python train_crosspoint.py --model dgcnn --dataset crosspoint_2dtm_2_new --epochs 250 --lr 0.001 --exp_name crosspoint_2dtm_2_onTheFly_2classes_simclr --batch_size 128 --print_freq 20 --k 15 --save_freq 10 --noise2d 0.0