
import argparse
import os
import time
from os.path import join
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import utils as utils
from datasets.sythetic_reflection import GeneralDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import random 
import cv2
import numpy as np

from  loss.functions import *

parser = argparse.ArgumentParser(description="Official Code for Zero-shor Learning for 360-degree Images")
parser.add_argument('--saving_dir', type=str, default="./results", help='Path of the validation dataset')
parser.add_argument('--data_dir', type=str, default="./", help='Path of the validation dataset')
parser.add_argument('--omega1',   type=float, default=10.0, help='omega1')
parser.add_argument('--omega2',   type=float, default=3.0, help='omega2')
parser.add_argument('--omega3',   type=float, default=5.0, help='omega3')
parser.add_argument('--omega4',   type=float, default=50.0, help='omega4')
parser.add_argument('--trans_lr',   type=float, default=0.000005, help='learning rate')
parser.add_argument('--refle_lr',   type=float, default=1e-3, help='learning rate')


opt = parser.parse_args()

saving_dir = join(opt.saving_dir, opt.data_dir.split('/')[-2], opt.data_dir.split('/')[-1][:-4],
                'omega-%d-%d-%d-%d' % (opt.omega1,opt.omega2,opt.omega3,opt.omega4))

if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.isfile('pre-processing/%s.pth' % (opt.data_dir.split('/')[-1][:-4])):
    ## ----------------------------------------------------------------------
    ## ----- Load image data ----------------------------------------------
    ## ----------------------------------------------------------------------
    ShiftGrids = []
    shifts = np.linspace(-0.1, 0.1, 5)
    X, Y= torch.meshgrid(torch.linspace(-1, 1,512), torch.linspace(-1, 1,512))
    for x in shifts:
        for y in shifts:
            grid = torch.cat((Y.unsqueeze(2)+y, X.unsqueeze(2)+x), 2)
            grid = grid.unsqueeze(0)
            ShiftGrids.append(grid)
    ShiftGrids = torch.cat(ShiftGrids, 0)
    ShiftGrids = ShiftGrids.cuda()

    Pano = load_test_data4(opt.data_dir).float()
    Glass_xyz = getGlassRegionXYZ(1.0, 512)
    Glass = gatherColorbyXYZ(Glass_xyz, Pano)
    Glass = Glass ** 0.5
    Glass = Glass.cuda()



    Hm = getHouseholderMatrix([1.0, 0.0, 0.0, -0.1])
    dists = 1.0/np.linspace(1,0.005, 50)


    ReferCandidateColor = []
    ReferCandidateXYZ = []
    GlassScaledXYZ = []

    for dist in dists:
        reflected_glass_candidate_xyz, glass_scaled_xyz = reflecting_xyz(Glass_xyz, dist, Hm, return_scale=True)
        GlassScaledXYZ.append(glass_scaled_xyz)
        ReferCandidateXYZ.append(reflected_glass_candidate_xyz)
        refer = gatherColorbyXYZ(reflected_glass_candidate_xyz, Pano)
        ReferCandidateColor.append(refer)
        
    GlassScaledXYZ = torch.stack(GlassScaledXYZ, dim=0).cuda() # N x 3 x H x W
    ReferCandidateXYZ = torch.stack(ReferCandidateXYZ, dim=0).cuda() # N x 3 x H x W
    ReferCandidateColor = torch.cat(ReferCandidateColor, dim=0).cuda()

    refer_xyz = ReferCandidateXYZ[0].detach().clone().cpu()
    Refer = ReferCandidateColor[[0]].detach().clone()
    GlassCandidateXYZ = []
    ReferScaledXYZ = []
    for dist in dists:
        reflected_refer_candidate_xyz, refer_scaled_xyz = reflecting_xyz(refer_xyz, dist, Hm, return_scale=True)
        GlassCandidateXYZ.append(reflected_refer_candidate_xyz)
        ReferScaledXYZ.append(refer_scaled_xyz)
    GlassCandidateXYZ = torch.stack(GlassCandidateXYZ, dim=0).cuda()
    ReferScaledXYZ = torch.stack(ReferScaledXYZ, dim=0).cuda()

    
    NN_index = torch.min((ReferCandidateXYZ.unsqueeze(1).cpu() - ReferScaledXYZ.unsqueeze(0).cpu()).norm(dim=2,keepdim=True), dim=1)[1]
    NN_index = NN_index.cuda()

    ## Bright Map 
    Bright = torch.max(Glass, dim=1, keepdim=True)[0]

    torch.save([ShiftGrids.cpu(),
                Pano.cpu(),
                Glass_xyz.cpu(),
                Glass.cpu(),
                Hm.cpu(),
                GlassScaledXYZ.cpu(),
                ReferCandidateXYZ.cpu(),
                ReferCandidateColor.cpu(),
                Refer.cpu(),
                GlassCandidateXYZ.cpu(),
                ReferScaledXYZ.cpu(),
                Bright.cpu(),
                NN_index.cpu(),
                ], 'pre-processing/%s.pth' % (opt.data_dir.split('/')[-1][:-4]))

[ShiftGrids,
Pano,
Glass_xyz,
Glass,
Hm,
GlassScaledXYZ,
ReferCandidateXYZ,
ReferCandidateColor,
Refer,
GlassCandidateXYZ,
ReferScaledXYZ,
Bright,
NN_index,
] = torch.load('pre-processing/%s.pth' % (opt.data_dir.split('/')[-1][:-4]))
ShiftGrids = ShiftGrids.cuda()
Pano = Pano.cuda()
Glass_xyz = Glass_xyz.cuda()
Glass = Glass.cuda()
Hm = Hm.cuda()
GlassScaledXYZ = GlassScaledXYZ.cuda()
ReferCandidateXYZ = ReferCandidateXYZ.cuda()
ReferCandidateColor = ReferCandidateColor.cuda()
Refer = Refer.cuda()
GlassCandidateXYZ = GlassCandidateXYZ.cuda()
ReferScaledXYZ = ReferScaledXYZ.cuda()
Bright = Bright.cuda()
NN_index = NN_index.cuda()

## ----------------------------------------------------------------------
## ----- Load model and modules ------------------------------------------
## ----------------------------------------------------------------------
color_grad_layer = GradientConcatLayer()
refle_net = Generator5(64)
color_grad_layer.cuda()
refle_net.load_state_dict(torch.load('initial-epoch1.pth'))
refle_net.cuda()

## ----------------------------------------------------------------------
## ----- Image pre-processing  ------------------------------------------
## ----------------------------------------------------------------------
## Gradient map 
GlassGrad = color_grad_layer(Glass, single_scale=True)
ReferGrad = color_grad_layer(Refer, single_scale=True)
ReferCandidateGrad = color_grad_layer(ReferCandidateColor, single_scale=True)

## ----------------------------------------------------------------------
## ----- Main training process ------------------------------------------
## ----------------------------------------------------------------------

auto_branch = []
trans_generation_branch = []
refle_generation_branch = []
for key, value in refle_net.named_parameters():
    if key_checking(key, ['trans_generator']):
        trans_generation_branch.append(value)
    if key_checking(key, ['refle_generator']):
        refle_generation_branch.append(value)
    if key_checking(key, ['encoder',  'decoder']):
        auto_branch.append(value)

optimizer_auto =  optim.Adam(auto_branch, lr=1e-5, betas=(0.9,0.999), weight_decay=1e-9)   
optimizer_refle_generation = optim.Adam(refle_generation_branch, lr=opt.refle_lr, betas=(0.9,0.999), weight_decay=1e-9)
optimizer_trans_generation = optim.Adam(trans_generation_branch, lr=opt.trans_lr, betas=(0.9,0.999), weight_decay=1e-9)

for k in range(600):

    ## -------------------------------------------------------------------------------------
    ## ---- Auto-encoder -------------------------------------------------------------------
    ## -------------------------------------------------------------------------------------
    index = int(random.random() * len(ReferCandidateColor))
    refle_net.zero_grad()
    glass_auto = refle_net.auto_encoder(Glass)
    refer_auto = refle_net.auto_encoder(ReferCandidateColor[[index]])
    glass_auto_grad = color_grad_layer(glass_auto, single_scale=True) 
    refer_auto_grad = color_grad_layer(refer_auto, single_scale=True) 
        
    loss = F.mse_loss(glass_auto, Glass) + F.mse_loss(refer_auto, ReferCandidateColor[[index]]) \
            + F.mse_loss(glass_auto_grad, GlassGrad) + F.mse_loss(refer_auto_grad, ReferCandidateGrad[[index]])
            
    loss.backward()
    optimizer_auto.step()

    ## -------------------------------------------------------------------------------------
    ## ---- Reflection restoration ---------------------------------------------------------------------
    ## -------------------------------------------------------------------------------------
    refle_net.zero_grad()
    trans, refle, refer = refle_net(Glass, Refer)
    refer_grad = color_grad_layer(refer, single_scale=True)

    ## 1. Matching loss
    color_diff = torch.mean(torch.abs(refer-ReferCandidateColor), dim=1, keepdim=True)
    color_diff = neighboring(color_diff, 5)
    color_matching_index = torch.min(color_diff, dim=0, keepdim=True)[1]
    target_color = torch.gather(ReferCandidateColor, 0, color_matching_index.repeat(1,ReferCandidateColor.shape[1],1,1))

    grad_diff = torch.mean(torch.abs(torch.tanh(5*GlassGrad)-ReferCandidateGrad), dim=1, keepdim=True)
    grad_diff = neighboring(grad_diff, 5)
    grad_matching_index = torch.min(grad_diff, dim=0, keepdim=True)[1]
    target_grad = torch.gather(ReferCandidateGrad, 0, grad_matching_index.repeat(1,ReferCandidateGrad.shape[1],1,1))
    
    loss_refer_color = F.mse_loss(refer, target_color)    
    loss_refer_grad = F.mse_loss(refer_grad, target_grad)     
    
    ## 2. Synthesis Loss
    glass_resyn = refle_net.synthesis3(trans.detach(), refle)
    loss_resyn = criterion_restoration(glass_resyn, Glass, alpha=opt.omega1 , clip=True)

    # ## 3. Gradient exclusive loss 
    # loss_refer_grad_excl = criterion_exclusive_gradient_rev(refer, trans.detach(), 2)

    ## Total Loss  
    loss_refle =  opt.omega3 * loss_refer_color \
                + opt.omega4 * loss_refer_grad \
                + loss_resyn 
                
    loss_refle.backward()
    optimizer_refle_generation.step() 


    ## -------------------------------------------------------------------------------------
    ## ---- Transmission restoration -------------------------------------------------------
    ## -------------------------------------------------------------------------------------
    refle_net.zero_grad()
    trans, refle, refer = refle_net(Glass, Refer)
    glass_resyn = refle_net.synthesis3(trans, refle.detach())
    ## 1. Gradient exclusive loss
    loss_trans_grad_excl = criterion_exclusive_gradient_rev(trans, refer.detach(), 2)
    ## 2. Re-synthesis loss
    loss_resyn = criterion_restoration(glass_resyn, Glass, alpha=opt.omega1 , clip=True)
    ## Total Loss  
    loss =   loss_resyn +  opt.omega2 * loss_trans_grad_excl
    loss.backward()
    optimizer_trans_generation.step() 

    if k % 50 == 0:
        if k == 0 :
            os.system('rm -rf %s' % join(saving_dir, '*.png')) 
        with torch.no_grad():
            trans, refle, refer = refle_net(Glass, Refer)
            glass_resyn = refle_net.synthesis3(trans, refle)
            torchvision.utils.save_image(Glass, '%s/%04d-01_glass.png' % (saving_dir, k) )
            torchvision.utils.save_image(glass_resyn, '%s/%04d-01_glass2_syn.png' % (saving_dir, k) )
            torchvision.utils.save_image(trans, '%s/%04d-02_trans.png' % (saving_dir, k) )
            torchvision.utils.save_image(refle, '%s/%04d-03_refle.png' % (saving_dir, k) )
            torchvision.utils.save_image(refer, '%s/%04d-04_refer.png' % (saving_dir, k) )
                
                
torch.save(refle_net.cpu().state_dict(),  '%s/refle_net_final.pth' % (saving_dir))


