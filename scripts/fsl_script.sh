#!/bin/bash

cd /home/kithmini.herath/codes/CrossPoint

python eval_fewshot.py --model_path /hpc/projects/group.czii/kithmini.herath/crosspoint-pretrained-models/dgcnn_cls_best.pth --dataset modelnet40 --k_way 5 --m_shot 10 --n_runs 50