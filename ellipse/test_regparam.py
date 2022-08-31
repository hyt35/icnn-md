# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:01:37 2022

@author: hongy
"""
#from icnn import DenseICGN
#from denoising_nets_for_mcmc import ICNN
import numpy as np
from odl.tomo.analytic.filtered_back_projection import fbp_op
import torch
from torch._C import LoggerBase
import torch.nn as nn
import torch.autograd as autograd
import torchvision
import matplotlib.pyplot as plt
#from iunets import iUNet
import parse_import
import logging
from datetime import datetime
import torch.nn.functional as F
from models import ICNNCouple
from pathlib import Path
from dival import get_standard_dataset
from dival.datasets.fbp_dataset import get_cached_fbp_dataset
import tensorflow as tf
import os
import odl
from tqdm import tqdm
import torch_wrapper
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
import time

IMPL = 'astra_cuda'
device = 'cuda:0'
workdir = "/local/scratch/public/hyt35/icnn-md/ellipse/deconv_nomom"

checkpoint_dir = os.path.join(workdir, "checkpoints")
tb_dir = os.path.join(workdir, "tensorboard")
tf.io.gfile.makedirs(checkpoint_dir)
tf.io.gfile.makedirs(tb_dir)


args=parse_import.parse_commandline_args()

#%%
# Initialize data

bsize=1


dataset = get_standard_dataset('ellipses', impl=IMPL)
ray_trafo = dataset.get_ray_trafo(impl=IMPL)
fbp_op_odl = odl.tomo.fbp_op(ray_trafo)
fbp_op = torch_wrapper.OperatorModule(fbp_op_odl).to(device)

CACHE_FILES = {
    'train':
        ('./cache_ellipses_train_fbp.npy', None),
    'validation':
        ('./cache_ellipses_validation_fbp.npy', None)}

def zero_one_transform(sample):
    fbp, gt = sample
    # min_ = torch.min(fbp)
    # max_ = torch.max(fbp)
    # fbp = (fbp-min_)/(max_-min_)
    return fbp, gt

start = time.time()
cached_fbp_dataset = get_cached_fbp_dataset(dataset, ray_trafo, CACHE_FILES)
dataset.fbp_dataset = cached_fbp_dataset

dataset_train = dataset.create_torch_dataset(
    part='train', reshape=((1,) + dataset.space[0].shape,
                            (1,) + dataset.space[1].shape),
                transform = zero_one_transform)

dataset_validation = dataset.create_torch_dataset(
    part='validation', reshape=((1,) + dataset.space[0].shape,
                                (1,) + dataset.space[1].shape))

train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=bsize, pin_memory=True)

test_dataloader = torch.utils.data.DataLoader(dataset_validation, batch_size = bsize, pin_memory=True)



#%%
# Initialize models

# icnn_couple = ICNNCouple(device = device, stepsize_init = 0.01, num_iters = 10, stepsize_clamp = (0.001,0.1), imsize = 128*128)
# icnn_couple.init_fwd(num_in_channels=3, num_filters=60, kernel_dim=3, num_layers=3, strong_convexity = 0.1, dense_size = 200)
# icnn_couple.init_bwd(num_in_channels=3, num_filters=75, kernel_dim=3, num_layers=5, strong_convexity = 0.5, dense_size = 200)
end = time.time()
#%%
# if args.from_checkpoint is not None:
#     icnn_couple.load_state_dict(torch.load(args.from_checkpoint))

n_epochs = args.num_epochs
noise_level = 0.1
reg_param = 0
stepsize = 0.01

closeness_reg = 1.0
closeness_update_nepochs = 50
closeness_update_multi = 1.05
#%%
if __name__ == '__main__': 
    if args.train:
        # icnn_couple.train()
        # opt = torch.optim.Adam(icnn_couple.parameters(),lr=1e-5,betas=(0.9,0.99))

        for epoch in range(n_epochs):
            total_loss = 0
            total_fwdbwd = 0
            batch_loss = 0
            batch_fwdbwd = 0
            #for idx, (batch_, gt) in enumerate(train_dataloader):
            for batch_, gt in train_dataloader:
                print(batch_.shape)
                print(gt.shape)
                # add gaussian noise
                start = time.time()
                batch = batch_.to(device)
                batch_noisy = batch + noise_level*torch.randn_like(batch) # 5% gaussian noise
                fbp_batch_noisy = fbp_op(batch) # apply fbp to noisy

                end=time.time()
                print(end-start)

                # define objective functions
                def recon_err(img):
                    tv_h = torch.abs(img[:,:,1:,:]-img[:,:,:-1,:]).sum()
                    tv_w = torch.abs(img[:,:,:,1:]-img[:,:,:,:-1]).sum()
                    tv = (tv_h+tv_w)
                    fidelity = torch.pow((img-fbp_batch_noisy),2).sum()
                    return (fidelity + reg_param*tv)/2
                
                def recon_err_grad(img):
                    return autograd.grad(recon_err(img), img)[0]


                fwd = fbp_batch_noisy.requires_grad_()
                loss = 0
                start = time.time()
                for i in range(600):
                    fwdGrad = recon_err_grad(fwd)
                    fwd = fwd - stepsize*fwdGrad
                    print(i)
                end = time.time()
                print(end - start)
                
                min_ = torch.min(fwd)
                max_ = torch.max(fwd)
                print(min_)
                print(max_)
                # fwd = (fwd-min_)/(max_-min_)
                #image_grid = make_grid(samples, 1, padding=2)
                with tf.io.gfile.GFile(os.path.join(workdir, "samples.png"), "wb") as fout:
                    save_image(fwd, fout)
                with tf.io.gfile.GFile(os.path.join(workdir, "true.png"), "wb") as fout:
                    save_image(gt, fout)
                break
            break
