#!/bin/bash

ml anaconda/latest
ml cuda/12.1.1_530.30.02
ml cudnn/8.9.7.29_cuda12

conda activate crosspoint_2dtm

cd /home/kithmini.herath/codes/CL-based-particle-picking

# Traning CrossPoint for classification, epochs = 100
python train_crosspoint_Original.py --model dgcnn --epochs 2 --lr 0.001 --exp_name crosspoint_dgcnn_cls --batch_size 20 --print_freq 20 --k 15 --save_freq 10 --test True


# # Training CrossPoint for part-segmentation, epochs = 100
# python train_crosspoint.py --model dgcnn_seg --epochs 2 --lr 0.001 --exp_name crosspoint_dgcnn_seg --batch_size 20 --print_freq 200 --k 15


# Fine-tuning for part-segmentation, epochs=300
# python train_partseg.py --exp_name dgcnn_partseg --pretrained_path /hpc/projects/group.czii/kithmini.herath/crosspoint-pretrained-models/dgcnn_partseg_best.pth --batch_size 8 --k 40 --test_batch_size 8 --epochs 2
