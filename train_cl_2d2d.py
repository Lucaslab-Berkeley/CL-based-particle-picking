from __future__ import print_function
import os
import random
import argparse
import torch
import math
import numpy as np
import wandb
from lightly.loss.ntx_ent_loss import NTXentLoss
import time
from sklearn.svm import SVC

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet101
from torch.utils.data import DataLoader

from datasets.data import cryoEM_cl2d2d_loader
import datasets.data_utils as d_utils
from models.dgcnn import ResNet
from util import IOStream, AverageMeter
from vis_utils import *
from losses import SupConLoss


def _init_():
    # if not os.path.exists('results'):
    #     os.makedirs('/hpc/projects/group.czii/kithmini.herath/crosspoint-trained-models/results')
    if not os.path.exists('/global/scratch/users/kithminiherath/results/'+args.exp_name):
        os.makedirs('/global/scratch/users/kithminiherath/results/'+args.exp_name)
    if not os.path.exists('/global/scratch/users/kithminiherath/results/'+args.exp_name+'/'+'models'):
        os.makedirs('/global/scratch/users/kithminiherath/results/'+args.exp_name+'/'+'models')
        
def loop(img_model, loader, opt, criterion, epoch, args, type_ = "train", device="cuda"):
    losses = AverageMeter()

    if type_ == "train":
        img_model.train()
        print(f'Start training epoch: ({epoch}/{args.epochs})')
    elif type_ == "val":
        img_model.eval()
        print(f'Start validation')
        
    if type_ == "train":
        imid_reg = args.imid_reg
    elif type_ == "val":
        imid_reg = 1.0
    
    for i, (data_t1, data_t2, labels) in enumerate(loader):
        if args.test and i>2: break
        data_t1, data_t2, labels = data_t1.to(device), data_t2.to(device), labels.to(device)
        # print(data_t1.device, data_t2.device, imgs.device, labels.device, next(img_model.parameters()).device)
        batch_size = data_t1.size()[0]

        if type_ == "train":
            opt.zero_grad()
        data = torch.cat((data_t1, data_t2)) # stacking img batches: (20,1,300,300) and (20,1,300,300) --> (40,1,300,300)

        if type_ == "train":
            _, projection_head_feats = img_model(data) # img_feats: features from the feature extractor (before the projection head), projection_head_feats: features out of the projection head --> for the contrast loss
        elif type_ == "val":
            with torch.no_grad():
                _, projection_head_feats = img_model(data)

        # breaking the img features back to the original stacks after processing:
        img_t1_feats = projection_head_feats[:batch_size, :]
        img_t2_feats = projection_head_feats[batch_size: , :]

        if args.loss == "SimCLR":
            loss_imid = criterion(img_t1_feats, img_t2_feats)    # normalizes in the loss function   
        elif args.loss == "SupCon":
            img1_feats = F.normalize(img_t1_feats, dim=1)
            img2_feats = F.normalize(img_t2_feats, dim=1)
            features1 = torch.cat([img1_feats.unsqueeze(1), img2_feats.unsqueeze(1)], dim=1)
            loss_imid = criterion(features1, labels)

        total_loss = loss_imid
        
        if type_ == "train":
            total_loss.backward()
            opt.step()

        losses.update(total_loss.item(), batch_size)

        if i % args.print_freq == 0 and type_ == "train":
            print('Epoch (%d), Batch(%d/%d), loss: %.6f' % (epoch, i, len(loader), losses.avg))
            
    return losses

def train(args):
    
    # wandb.init(project="CrossPoint", name=args.exp_name)
    noise2d = float_list(args.noise2d)
    
    # the following is an image transform
    transform = transforms.Compose([transforms.ToTensor(), # This converts PIL Image to tensor and scales to [0, 1]
                                # nn.AvgPool2d(kernel_size=2, stride=2),
                                d_utils.AddGaussianNoise_multiple(stds=noise2d, p=1),
                                d_utils.ProjectionRotate(angles=[90, 180, 270], p=0.5), # MiLoPYP paper mentions that for 3D volumes rotations of multiples of 90 avoids interpolation artifacts and simplifies the resampling of voxel values
                                # transforms.Resize((224, 224), antialias=True),
                                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                # transforms.CenterCrop((300, 300)),
                                transforms.RandomHorizontalFlip(), # default probability = 0.5 -- MiLOPYP does random horizontal and vertical flip as well (https://github.com/nextpyp/cet_pick/blob/426cc92d9f5459ec4fb9ae91b2bddac65363f870/cet_pick/datasets/tomo_post_proj_angle_select.py#L40)
                                transforms.RandomVerticalFlip(), # from MiLOPYP 
                                transforms.RandomResizedCrop(args.image_size, (0.7,1.0),(1.0,1.0)) # from MiLOPYP 
                                # transforms.Normalize((0.5,), (0.5,)) # normalize within -1,1 -- removed this due to the simulated and real data already being standard normalized and didn't want to cause too much distortion by renormalization
                                # transforms.v2.GaussianNoise(mean=0.0, sigma=0.05, clip=True),
                                ])

    #### Edit for train-test split
    data_dir = f"/global/scratch/users/kithminiherath/{args.dataset}"
    type_ = "train"
    train_set = cryoEM_cl2d2d_loader(f"{data_dir}/{type_}", img_transform = transform, type_=type_, filter_type=args.filter_type, filter_cutoff=args.filter_cutoff, fourier_transformed=args.fourier_transformed)
    train_loader = DataLoader(train_set, num_workers=12, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True) # the n_imgs parameter is not being used as of now
    
    type_ = "test"
    val_set = cryoEM_cl2d2d_loader(f"{data_dir}/{type_}", img_transform = transform, type_=type_, filter_type=args.filter_type, filter_cutoff=args.filter_cutoff, fourier_transformed=args.fourier_transformed)
    val_loader = DataLoader(val_set, num_workers=12, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)

    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)
    
    if args.test:
        print("Testing Phase !!!")

    #Try to load models
    if args.model == 'resnet50':
        img_model = ResNet(resnet50(), feat_dim = 2048) # i think this is loading from pretrained weights. but i'm replacing the first layer with my own layer...so not sure how much this affects the learned behavior of resnets
    # the resnet50 will take an image of any size and give an output based on the feat_dim argument size
    # the inv model (projection head) is set to take in a 2048 dim feature and output 256 dim feature for the contrast loss -- so would need to change this in the future along with feat_dim above to get what you want as feature dimensions
    elif args.model == 'resnet101':
        img_model = ResNet(resnet101(), feat_dim = 2048)
    
    img_model = img_model.to(device)
        
    parameters = img_model.parameters()

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=1e-6)
    else:
        print("Use Adam")
        opt = optim.Adam(parameters, lr=args.lr, weight_decay=1e-6)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0, last_epoch=-1)
    
    if args.loss == "SimCLR":
        criterion = NTXentLoss(temperature = 0.1).to(device)
    elif args.loss == "SupCon":
        print("Using supervised contrastive loss ***")
        criterion = SupConLoss(temperature = 0.1).to(device)
    
    best_acc = 0
    train_loss_lst, val_loss_lst, train_acc_lst, val_acc_lst = [], [], [], []
    start_train_loop_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        ####################
        # Train
        ####################
        train_losses = loop(img_model, train_loader, opt, criterion, epoch, args, type_ = "train", device=device)
        lr_scheduler.step()
        
        outstr = 'Train %d, loss: %.6f' % (epoch, train_losses.avg)
        train_loss_lst.append(train_losses.avg)
        ###### Plot train curves
        print(outstr) 
        
        ####################
        # Validation
        ####################
        val_losses = loop(img_model, val_loader, opt, criterion, epoch, args, type_ = "val", device=device)
                
        outstr = 'Val %d, loss: %.6f' % (epoch, val_losses.avg)
        val_loss_lst.append(val_losses.avg)
        ###### Plot train curves
        print(outstr) 
        
        # Fit classifier on train set
        model_tl = SVC(C = 0.1, kernel ='linear')
        feats_train, labels_train = [], []
        for i, (imgs, _, label) in enumerate(train_loader):
            if args.test and i>2: 
                break
            elif i > 80:
                break # only using a part of the train set for the classification
                
            labels = label.numpy().tolist()
            data = imgs.to(device)
            with torch.no_grad():
                feats, _ = img_model(data)
            feats = feats.detach().cpu().numpy()
            feats_train.extend(feats)
            labels_train += labels
        feats_train = np.array(feats_train)
        labels_train = np.array(labels_train)

        model_tl.fit(feats_train, labels_train) 
        train_acc_lst.append(model_tl.score(feats_train, labels_train))
        
        # Testing/ validation classifier
        feats_val, labels_val = [], []
        for i, (imgs, _, label) in enumerate(val_loader):
            if args.test and i>2: break
            labels = label.numpy().tolist()
            data = imgs.to(device)
            with torch.no_grad():
                feats,_ = img_model(data)
            feats = feats.detach().cpu().numpy()
            feats_val.extend(feats)
            labels_val += labels
        feats_val = np.array(feats_val)
        labels_val = np.array(labels_val)
        
        val_accuracy = model_tl.score(feats_val, labels_val)
        val_acc_lst.append(val_accuracy)
        print(f"Linear Accuracy (val): {val_accuracy}")
        
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            print('==> Saving Best Model...')
            save_img_model_file = os.path.join(f'/global/scratch/users/kithminiherath/results/{args.exp_name}/models/',
                         'img_model_best.pth')
            torch.save(img_model.state_dict(), save_img_model_file)
  
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            save_img_model_file = os.path.join(f'/global/scratch/users/kithminiherath/results/{args.exp_name}/models/',
                         'img_model_ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(img_model.state_dict(), save_img_model_file)
            
            plot_train_losses(train_loss_lst, val_loss_lst, return_fig= True)
            plt.savefig(f'/global/scratch/users/kithminiherath/results/{args.exp_name}/train_losses_latest.png')

            plot_classifier_acc(train_acc_lst, val_acc_lst, return_fig= True)
            plt.savefig(f'/global/scratch/users/kithminiherath/results/{args.exp_name}/classifier_accuracies_latest.png')


    print(f"Total train time = {(time.time() - start_train_loop_time)/60} minutes")
    print('==> Saving Last Model...')
    save_img_model_file = os.path.join(f'/global/scratch/users/kithminiherath/results/{args.exp_name}/models/',
                         'img_model_last.pth')
    torch.save(img_model.state_dict(), save_img_model_file)
    plot_train_losses(train_loss_lst, val_loss_lst, return_fig= True)
    plt.savefig(f'/global/scratch/users/kithminiherath/results/{args.exp_name}/train_losses_last.png')
    
    plot_classifier_acc(train_acc_lst, val_acc_lst, return_fig= True)
    plt.savefig(f'/global/scratch/users/kithminiherath/results/{args.exp_name}/classifier_accuracies_last.png')

def float_list(arg):
    return list(map(float, arg.split(',')))

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='resnet50', metavar='N',
                        choices=['resnet50', 'resnet101'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--image_size', type=int, metavar='image_size',
                        help='Size of image)')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action="store_true", help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    parser.add_argument('--test', type=bool, default=False, help='whether to test the code: True - testing')
    parser.add_argument('--loss', type=str, default="SimCLR", help='SimCLR: Contrastive Loss, SupCon: Supervised Contrastive Loss')
    parser.add_argument('--load_perc', type=float, default=1.0, help='The percentage of data to be loaded onto the RAM')
    parser.add_argument('--dataset', type=str, help='crosspoint_2dtm_2_new, crosspoint_2dtm_4_new')
    parser.add_argument('--noise2d', type=str, help='comma-separated list of std values for noise (e.g., 0.05 0.1 0.15)')
    parser.add_argument('--noise3d', type=float, default=0.1, help='the std of the noise that should be added to the templates (3d)')
    parser.add_argument('--imid_reg', type=float, default=1.0, help='regularization parameter for the IMID loss')
    parser.add_argument('--filter_type', type=str, help='Type of filter: lowpass, highpass, "", None')
    parser.add_argument('--filter_cutoff', type=float, default=0.1, help='Normalized cutoff frequency of the filter')
    parser.add_argument('--fourier_transformed', type=bool, default=False, help='Fourier transformed? True - will be fourier transformed post filtering')
    args = parser.parse_args()

    _init_()

    print(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        print('Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        print('Using CPU')

    if not args.eval:
        train(args)


