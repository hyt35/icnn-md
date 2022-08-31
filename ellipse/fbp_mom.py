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
from models import ICNNCoupleMomentum
from pathlib import Path
from dival import get_standard_dataset
from dival.datasets.fbp_dataset import get_cached_fbp_dataset
import tensorflow as tf
import os
import odl
from tqdm import tqdm
import torch_wrapper
from torch.utils import tensorboard


IMPL = 'astra_cuda'
device = 'cuda:0'
workdir = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/fbp_mom"

checkpoint_dir = os.path.join(workdir, "checkpoints")
tb_dir = os.path.join(workdir, "tensorboard")
tf.io.gfile.makedirs(checkpoint_dir)
tf.io.gfile.makedirs(tb_dir)

writer = tensorboard.SummaryWriter(tb_dir)

logging.basicConfig(filename=os.path.join(workdir,datetime.now().strftime("%d-%m_%H-%M-fbp_mom.log")), filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger()

args=parse_import.parse_commandline_args()

#%%
# Initialize data

bsize=200


dataset = get_standard_dataset('ellipses', impl=IMPL)
ray_trafo = dataset.get_ray_trafo(impl=IMPL)
fbp_op_odl = odl.tomo.fbp_op(ray_trafo)
fbp_op = torch_wrapper.OperatorModule(fbp_op_odl).to(device)

CACHE_FILES = {
    'train':
        ('./cache_ellipses_train_fbp.npy', None),
    'validation':
        ('./cache_ellipses_validation_fbp.npy', None)}

# def zero_one_transform(sample):
#     fbp, gt = sample
#     min_ = torch.min(fbp)
#     max_ = torch.max(fbp)
#     fbp = (fbp-min_)/(max_-min_)
#     return fbp, gt


cached_fbp_dataset = get_cached_fbp_dataset(dataset, ray_trafo, CACHE_FILES)
dataset.fbp_dataset = cached_fbp_dataset

dataset_train = dataset.create_torch_dataset(
    part='train', reshape=((1,) + dataset.space[0].shape,
                            (1,) + dataset.space[1].shape))

dataset_validation = dataset.create_torch_dataset(
    part='validation', reshape=((1,) + dataset.space[0].shape,
                                (1,) + dataset.space[1].shape))

train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=bsize)

test_dataloader = torch.utils.data.DataLoader(dataset_validation, batch_size = bsize)
#%%
# Initialize models

icnn_couple = ICNNCoupleMomentum(device = device, stepsize_init = 0.01, num_iters = 10, stepsize_clamp = (0.001,0.1))
icnn_couple.init_fwd(num_in_channels=1, num_filters=60, kernel_dim=3, num_layers=2, strong_convexity = 0.1, dense_size = 20)
icnn_couple.init_bwd(num_in_channels=1, num_filters=70, kernel_dim=3, num_layers=2, strong_convexity = 0.1, dense_size = 20)
#%%
logger.info("fwd params"+str( sum(p.numel() for p in icnn_couple.fwd_model.parameters())))
logger.info("bwd params"+str( sum(p.numel() for p in icnn_couple.bwd_model.parameters())))
#%%
if args.from_checkpoint is not None:
    icnn_couple.load_state_dict(torch.load(args.from_checkpoint))

n_epochs = args.num_epochs
noise_level = 0.1
reg_param = 0.15
stepsize = 0.01

closeness_reg = 1.0
closeness_update_nepochs = 50
closeness_update_multi = 1.05
#%%
if __name__ == '__main__': 
    if args.train:
        icnn_couple.train()
        opt = torch.optim.Adam(icnn_couple.parameters(),lr=1e-5,betas=(0.9,0.99))

        for epoch in range(n_epochs):
            total_loss = 0
            total_fwdbwd = 0
            batch_loss = 0
            batch_fwdbwd = 0
            for idx, (batch_, _) in enumerate(train_dataloader):
                # add gaussian noise
                batch = batch_.to(device)
                batch_noisy = batch + noise_level*torch.randn_like(batch) # 5% gaussian noise
                fbp_batch_noisy = fbp_op(batch) # apply fbp to noisy



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

                # for stepsize in icnn_couple.stepsize:
                #     fwdGrad = recon_err_grad(fwd)
                #     fwd = icnn_couple.bwd_model(icnn_couple.fwd_model(fwd) - stepsize*fwdGrad) 
                #     loss += recon_err(fwd)

                iterates = icnn_couple(fwd, recon_err_grad)
                for i in iterates:
                    loss += recon_err(i)
                fwd = iterates[-1]
                #loss = recon_err(fwd)
                closeness1 = icnn_couple.fwdbwdloss(fbp_batch_noisy)
                closeness2 = icnn_couple.fwdbwdloss(fwd)
                closeness3 = icnn_couple.fwdbwdloss(batch_.to(device))

                err = loss+(closeness1+closeness2+closeness3)*closeness_reg
                
                opt.zero_grad()
                err.backward()
                opt.step()
                
                icnn_couple.clip_fwd()
                icnn_couple.clip_bwd()
                icnn_couple.clamp_stepsizes()
                
                total_loss += err.item()
                total_fwdbwd += closeness1.item() + closeness2.item() + closeness3.item()
                batch_loss += err.item()
                batch_fwdbwd += closeness1.item() + closeness2.item() + closeness3.item()
                if(idx % args.num_batches == args.num_batches-1):
                    avg_loss = batch_loss/args.num_batches/bsize
                    avg_fwdbwd = batch_fwdbwd/args.num_batches/bsize
                    print("loss", loss.item(), "closeness masked", closeness1.item(), "closeness inpaint", closeness2.item(), "closeness true", closeness3.item())
                    train_log = "epoch:[{}/{}] batch:[{}/{}], avg_loss = {:.4f}, avg_fwdbwd = {:.4f}".\
                      format(epoch+1, args.num_epochs, idx+1, len(train_dataloader), avg_loss, avg_fwdbwd)
                    print(train_log)
                    logger.info(train_log)

                    batch_loss = 0
                    batch_fwdbwd = 0
            
            print("Epoch", epoch, "total loss", total_loss, "fwdbwd", total_fwdbwd)
            writer.add_scalar("train_avg_loss", total_loss, epoch)
            writer.add_scalar("train_avg_fwdbwd", total_fwdbwd, epoch)
            # Checkpoint
            # Increase closeness regularization
            if (epoch%closeness_update_nepochs == closeness_update_nepochs-1):
                closeness_reg = closeness_reg*closeness_update_multi

            if (epoch%args.checkpoint_freq == args.checkpoint_freq-1):
                torch.save(icnn_couple.state_dict(), os.path.join(checkpoint_dir, str(epoch+1)))
            # Log
                logger.info("\n====epoch:[{}/{}], epoch_loss = {:.2f}, epoch_fwdbwd = {:.4f}====\n".format(epoch+1, args.num_epochs, total_loss, total_fwdbwd))
