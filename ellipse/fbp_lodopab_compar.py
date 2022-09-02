# -*- coding: utf-8 -*-
"""
Created on Fri Sep  13:44:37 2022

@author: hongy
"""
#from icnn import DenseICGN
#from denoising_nets_for_mcmc import ICNN
from sqlite3 import TimeFromTicks
from xxlimited import foo
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
from models import ICNNCoupleMomentum, ICNNCouple
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
from odl.ufunc_ops.ufunc_ops import log_op
from dival.evaluation import TaskTable
from dival.measure import PSNR, SSIM
from dival.reconstructors.odl_reconstructors import FBPReconstructor
from dival.datasets.standard import get_standard_dataset
from dival.util.constants import MU_MAX

IMPL = 'astra_cuda'
device0 = 'cuda:1'
device1 = 'cuda:2'

workdir_mom = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/fbp_mom"
workdir_nomom = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/fbp_nomom"


checkpoint_dir_mom = os.path.join(workdir_mom, "checkpoints", "20")
checkpoint_dir_nomom = os.path.join(workdir_nomom, "checkpoints", "20")
figs_dir = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/figs/lodopab"
args=parse_import.parse_commandline_args()



#%%
# Load models

icnn_couple_mom = ICNNCoupleMomentum(device = device0, stepsize_init = 0.01, num_iters = 10, stepsize_clamp = (0.001,0.1))
icnn_couple_mom.init_fwd(num_in_channels=1, num_filters=60, kernel_dim=3, num_layers=2, strong_convexity = 0.1, dense_size = 20)
icnn_couple_mom.init_bwd(num_in_channels=1, num_filters=70, kernel_dim=3, num_layers=2, strong_convexity = 0.1, dense_size = 20)
icnn_couple_mom.load_state_dict(torch.load(checkpoint_dir_mom, map_location=device0))

icnn_couple_nomom = ICNNCouple(device = device1, stepsize_init = 0.01, num_iters = 10, stepsize_clamp = (0.001,0.1))
icnn_couple_nomom.init_fwd(num_in_channels=1, num_filters=60, kernel_dim=3, num_layers=2, strong_convexity = 0.1, dense_size = 20)
icnn_couple_nomom.init_bwd(num_in_channels=1, num_filters=70, kernel_dim=3, num_layers=2, strong_convexity = 0.1, dense_size = 20)
icnn_couple_nomom.load_state_dict(torch.load(checkpoint_dir_nomom, map_location = device1))

#print("fwd params"+str( sum(p.numel() for p in icnn_couple_mom.bwd_model.parameters())))
# 
#%%
# Initialize data

bsize=10


dataset = get_standard_dataset('lodopab', impl=IMPL)
ray_trafo = dataset.get_ray_trafo(impl=IMPL)
fbp_op_odl = odl.tomo.fbp_op(ray_trafo)
fbp_op = torch_wrapper.OperatorModule(fbp_op_odl).to(device0)

# fbp_reconstructor = FBPReconstructor(
#     ray_trafo, hyper_params={
#         'filter_type': 'Hann',
#         'frequency_scaling': 0.8}).to(device0)
    
CACHE_FILES = {
    'train':
        ('./cache_lodopab_train_fbp.npy', None),
    'validation':
        ('./cache_lodopab_validation_fbp.npy', None)}

def zero_one_transform(fbp):
    foo = fbp.view(fbp.shape[0], -1)
    min_ = torch.min(foo, dim = 1)[0]
    max_ = torch.max(foo, dim = 1)[0]

    min__ = min_[:,None,None,None]
    max__ = max_[:,None,None,None]
    fbp = (fbp-min__)/(max__-min__)
    return fbp


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
noise_level = 0.1
reg_param = 0.15
stepsize = 0.01

#%%
resize_op = torchvision.transforms.Resize([128,128])
if __name__ == '__main__': 
    icnn_couple_mom.eval()
    icnn_couple_nomom.eval()
    total_loss = 0
    total_fwdbwd = 0
    batch_loss = 0
    batch_fwdbwd = 0
    
    fig, ax = plt.subplots(figsize = (8,6)) # loss plot
    fig2, ax2 = plt.subplots(figsize = (8,6)) # fwdbwd plot
    for idx, (batch_, gt) in enumerate(test_dataloader):
        # add gaussian noise
        gt = resize_op(gt)
        with tf.io.gfile.GFile(os.path.join(figs_dir, "gt.png"), "wb") as fout:
            save_image(gt, fout)
        
        batch = batch_.to(device0)
        batch_noisy = batch + noise_level*torch.randn_like(batch) # 5% gaussian noise

        # print("batch shape", batch.shape)
        # #foo = fbp_op(batch)
        # foo = fbp_op(batch)
        # print(foo.shape)
        # print(batch.shape)
        # print("max", torch.max(foo), "min", torch.min(foo))
        # plt.imshow(foo[0,0,:,:].cpu().numpy())
        # plt.colorbar()
        # plt.savefig(os.path.join(figs_dir, "foo.png"))


        fbp_batch_noisy = resize_op(zero_one_transform(fbp_op(batch))) # apply fbp to noisy
        #print(fbp_batch_noisy.shape)
        fbp_batch_noisy1 = fbp_batch_noisy.to(device1)
        
        
        print("max", torch.max(fbp_batch_noisy), "min", torch.min(fbp_batch_noisy))
        with tf.io.gfile.GFile(os.path.join(figs_dir, "noisy.png"), "wb") as fout:
            save_image(fbp_batch_noisy, fout)
        # define objective functions
        def recon_err(img):
            tv_h = torch.abs(img[:,:,1:,:]-img[:,:,:-1,:]).sum()
            tv_w = torch.abs(img[:,:,:,1:]-img[:,:,:,:-1]).sum()
            tv = (tv_h+tv_w)
            fidelity = torch.pow((img-fbp_batch_noisy),2).sum()
            return (fidelity + reg_param*tv)/2
        
        def recon_err_grad(img):
            return autograd.grad(recon_err(img), img)[0]

        def recon_err1(img):
            tv_h = torch.abs(img[:,:,1:,:]-img[:,:,:-1,:]).sum()
            tv_w = torch.abs(img[:,:,:,1:]-img[:,:,:,:-1]).sum()
            tv = (tv_h+tv_w)
            fidelity = torch.pow((img-fbp_batch_noisy1),2).sum()
            return (fidelity + reg_param*tv)/2
        
        def recon_err_grad1(img):
            return autograd.grad(recon_err1(img), img)[0]
        
        init_err = recon_err(fbp_batch_noisy).item()
        # DEVICE 0: MOMENTUM
        start = time.time()
        fwd = fbp_batch_noisy.clone().detach().requires_grad_()
        loss_mom = [init_err]
        closeness_mom = []
        iterates = icnn_couple_mom(fwd, recon_err_grad)
        for i in iterates:
            loss = recon_err(i).item()
            loss_mom.append(loss)
            closeness = icnn_couple_mom.fwdbwdloss(i).item()
            closeness_mom.append(closeness)
        end = time.time()
        print("Momentum time", end-start)

        with tf.io.gfile.GFile(os.path.join(figs_dir, "adapmd_mom.png"), "wb") as fout:
            save_image(iterates[-1], fout)

        ax.plot(loss_mom, label = "adaptive MD mom")
        ax2.plot(closeness_mom, label = "adaptive MD mom")      
        
        print("Momentum SS:", icnn_couple_mom.stepsize)
        # DEVICE 1: NO MOMENTUM
        start = time.time()
        fbp_batch_noisy1 = fbp_batch_noisy.clone().detach().to(device1)
        fwd = fbp_batch_noisy1.clone().detach().requires_grad_()
        loss_nomom = [init_err]
        closeness_nomom = []
        iterates = icnn_couple_nomom(fwd, recon_err_grad1)
        for i in iterates:
            loss = recon_err1(i).item()
            loss_nomom.append(loss)
            closeness = icnn_couple_nomom.fwdbwdloss(i).item()
            closeness_nomom.append(closeness)

        end = time.time()
        print("No momentum time", end-start)

        print("No momentum SS:", icnn_couple_nomom.stepsize)

        ax.plot(loss_nomom, label = "adaptive MD nomom")
        ax2.plot(closeness_nomom, label = "adaptive MD nomom")
        with tf.io.gfile.GFile(os.path.join(figs_dir, "adapmd_nomom.png"), "wb") as fout:
            save_image(iterates[-1], fout)

        # DEVICE 1: NO MOMENTUM (FIX STEPSIZE)
        start = time.time()
        fbp_batch_noisy1 = fbp_batch_noisy.clone().detach().to(device1)
        fwd = fbp_batch_noisy1.clone().detach().requires_grad_()
        loss_nomom = [init_err]
        closeness_nomom = []
        for i in range(10):
            ss = 0.1/(i+1)

            fwdGrad = recon_err_grad1(fwd)
            fwd = icnn_couple_nomom.bwd_model(icnn_couple_nomom.fwd_model(fwd) - ss*fwdGrad).detach()

            loss = recon_err1(fwd).item()
            loss_nomom.append(loss)
            closeness = icnn_couple_nomom.fwdbwdloss(fwd).item()
            closeness_nomom.append(closeness)

        end = time.time()
        print("No momentum fix ss time", end-start)

        #print("No momentum SS:", icnn_couple_nomom.stepsize)

        ax.plot(loss_nomom, label = "fixss MD nomom")
        ax2.plot(closeness_nomom, label = "fixss MD nomom")
        with tf.io.gfile.GFile(os.path.join(figs_dir, "fixmd_nomom.png"), "wb") as fout:
            save_image(iterates[-1], fout)


        # COMPARISON: GD
        print("Start GD")
        start = time.time()
        fwd = fbp_batch_noisy.clone().detach().requires_grad_()
        
        for ss in [0.001,0.01,0.05,0.1]:
            gd_loss = [init_err]
            for i in range(10):
                fwdGrad = recon_err_grad(fwd)
                fwd = fwd - ss*fwdGrad
                gd_loss.append(recon_err(fwd).item())
            # plot
            ax.plot(gd_loss, label = "GD "+str(ss), marker = 'x')
    
            with tf.io.gfile.GFile(os.path.join(figs_dir, "gd"+str(ss)+".png"), "wb") as fout:
                save_image(fwd, fout)
        end = time.time()
        print("Total GD time", end-start)

        # COMPARISON: ADAM
        print("Start Adam")
        start=time.time()
        for lr in [0.001,0.01,0.05,0.1]:
            adam_loss = []
            par = fbp_batch_noisy.clone().detach().requires_grad_()
            par.requires_grad_()
            optimizer = torch.optim.Adam([par], lr=lr)
            for i in range(10+1):
                optimizer.zero_grad()
                loss = recon_err(par)
                loss.backward(retain_graph=True)
                optimizer.step()
                
                #print("adam recon", recon_err(par).item())
                adam_loss.append(loss.item())
            ax.plot(adam_loss, label = "adam "+str(lr), marker = 'o')

            with tf.io.gfile.GFile(os.path.join(figs_dir, "adam"+str(lr)+".png"), "wb") as fout:
                save_image(fwd, fout)
        end = time.time()
        print("Total Adam time", end-start)


        # Plotter
    
        fig.legend()
        fig2.legend()

        fig.savefig(os.path.join(figs_dir, "losses"))
        fig2.savefig(os.path.join(figs_dir, "fwdbwd"))
        break