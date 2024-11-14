#!/bin/bash

ml anaconda/latest
ml cuda/12.1.1_530.30.02
ml cudnn/8.9.7.29_cuda12

conda activate crosspoint_2dtm

cd /home/kithmini.herath/codes/CL-based-particle-picking

# Training CrossPoint for classification, epochs = 100
python train_crosspoint.py --model dgcnn --dataset crosspoint_2dtm_2_new --epochs 2 --lr 0.001 --exp_name test_2classes --batch_size 5 --print_freq 50 --k 15 --test True --save_freq 10 --load_perc 0.2 --noise2d 0.0 --loss SupCon