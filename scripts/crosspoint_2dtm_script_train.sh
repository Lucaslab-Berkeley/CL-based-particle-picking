#!/bin/bash

cd /home/kithmini.herath/codes/CrossPoint

# crosspoint_2dtm_2 : val_included 0
# crosspoint_2dtm_4 : val_inlcuded 1
# Traning CrossPoint for classification, epochs = 100, --val_included 1
python train_crosspoint.py --model dgcnn --dataset crosspoint_2dtm_4 --val_included 1 --epochs 200 --lr 0.001 --exp_name crosspoint_2dtm_4_supcon_2dNoise0.0_3dNoise0.01 --batch_size 128 --print_freq 20 --k 15 --save_freq 10 --noise3d 0.01 --noise2d 0.0 --loss SupCon