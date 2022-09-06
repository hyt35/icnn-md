# -*- coding: utf-8 -*-
"""
Created on Fri Sep  13:44:37 2022

@author: hongy
"""
#from icnn import DenseICGN
#from denoising_nets_for_mcmc import ICNN
from pickle import TRUE
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
from dival import get_standard_dataset
from dival.datasets.fbp_dataset import get_cached_fbp_dataset
import tensorflow as tf
from models import ICNNCoupleMomentum, ICNNCouple
from pathlib import Path
import os
from tqdm import tqdm
import torch_wrapper
import time
import scipy.optimize as spopt
import odl
from dival.datasets.standard import get_standard_dataset

#%% EXPERIMENT PARAMETERS
IMPL = 'astra_cuda'
#MODE = "ELLIPSE"
MODE = "LODOPAB"
EXPTYPE = "MAN" # experiment type 
MAXITERS = 50 # maximum number of optim iter for experiments
PLOT = False # use the other script to plot.
#%% EXPERIMENT FLAGS
if EXPTYPE == "ALL":
    GRABNEW = True
    COMPUTETRUEMIN = True
    STEPSIZEEXT = True
    CONSTSS = True
    MAPTRANSFER = True
    DOMAINCHANGE = True
    
else: # manual
    GRABNEW = False
    COMPUTETRUEMIN = False
    STEPSIZEEXT = False
    CONSTSS = False
    MAPTRANSFER = True
    DOMAINCHANGE = False
    

#%% INITIALIZATION
device0 = 'cuda:1'
device1 = 'cuda:2'

workdir_mom = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/fbp_mom"
workdir_nomom = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/fbp_nomom"

checkpoint_dir_mom = os.path.join(workdir_mom, "checkpoints", "20")
checkpoint_dir_nomom = os.path.join(workdir_nomom, "checkpoints", "20")

if MODE == "ELLIPSE":
    figs_dir = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/figs/ellipse"
else:
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

# save variable for later
adap_ss_mom = icnn_couple_mom.stepsize
adap_ss_nomom = icnn_couple_nomom.stepsize
#%%
# Initialize datasets and fbp operator

#%% parameter
noise_level = 0.1
reg_param = 0.15
stepsize = 0.01

#%%
resize_op = torchvision.transforms.Resize([128,128])
if __name__ == '__main__': 
    icnn_couple_mom.eval()
    icnn_couple_nomom.eval()
    # fig, ax = plt.subplots(figsize = (8,6)) # loss plot
    # fig2, ax2 = plt.subplots(figsize = (8,6)) # fwdbwd plot
    #%% check whether to get new data sample
    if GRABNEW:
        bsize=10

        if MODE == "ELLIPSE":
            dataset = get_standard_dataset('ellipses', impl=IMPL)
            CACHE_FILES = {
                'train':
                    ('./cache_ellipses_train_fbp.npy', None),
                'validation':
                    ('./cache_ellipses_validation_fbp.npy', None)}

            def fbp_postprocess(fbp): # no correction needed for ellipses dataset
                return fbp
        else:
            dataset = get_standard_dataset('lodopab', impl=IMPL)
            CACHE_FILES = {
                'train':
                    ('./cache_lodopab_train_fbp.npy', None),
                'validation':
                    ('./cache_lodopab_validation_fbp.npy', None)}

            def fbp_postprocess(fbp): # correction for lodopab fbp inversion issue
                foo = fbp.view(fbp.shape[0], -1)
                min_ = torch.min(foo, dim = 1)[0]
                max_ = torch.max(foo, dim = 1)[0]

                min__ = min_[:,None,None,None]
                max__ = max_[:,None,None,None]
                fbp = (fbp-min__)/(max__-min__)
                return fbp

        # operators and dataset
        ray_trafo = dataset.get_ray_trafo(impl=IMPL)
        fbp_op_odl = odl.tomo.fbp_op(ray_trafo)
        fbp_op = torch_wrapper.OperatorModule(fbp_op_odl).to(device0)
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
        for idx, (batch_, gt) in enumerate(test_dataloader):
            gt = resize_op(gt)
            # add gaussian noise
            batch = batch_.to(device0)
            batch_noisy = batch + noise_level*torch.randn_like(batch) # 10% gaussian noise

            fbp_batch_noisy = resize_op(fbp_postprocess(fbp_op(batch))) # apply fbp to noisy
            break
        torch.save(gt, os.path.join("dat" + MODE, MODE+"gt.pt"))
        torch.save(fbp_batch_noisy, os.path.join("dat" + MODE, MODE+"batch.pt"))
        fbp_batch_noisy1 = fbp_batch_noisy.clone().to(device1)
    else:
        gt = torch.load(os.path.join("dat" + MODE, MODE+"gt.pt"))
        fbp_batch_noisy = torch.load(os.path.join("dat" + MODE, MODE+"batch.pt"))
        fbp_batch_noisy1 = fbp_batch_noisy.clone().to(device1)

    # define objective functions (on respective devices)
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
    init_closeness_mom = icnn_couple_mom.fwdbwdloss(fbp_batch_noisy).item()
    init_closeness_nomom = icnn_couple_nomom.fwdbwdloss(fbp_batch_noisy1).item()
    print("init complete")

    #%%
    if COMPUTETRUEMIN:
        # perform lots of gd

        fwd = fbp_batch_noisy.clone().detach().requires_grad_(True)
        for i in range(2000):
            fwd = fwd.detach().requires_grad_(True)
            fwdGrad = recon_err_grad(fwd)
            fwd = fwd - 5e-3*fwdGrad
        # plot
        true_min = recon_err(fwd).item()
        np.save(os.path.join("dat" + MODE, "true_min_val"), true_min)
        np.save(os.path.join("dat" + MODE, "true_min_arr"), fwd.detach().cpu().numpy())
        print("compute true min", true_min)
    
    true_min = np.load(os.path.join("dat" + MODE, "true_min_val.npy"))
    true_fwd = torch.from_numpy(np.load(os.path.join("dat" + MODE, "true_min_arr.npy"))).to(device0)
    true_fwd1 = true_fwd.clone().to(device1)
    init_l2 = torch.linalg.vector_norm(true_fwd - fbp_batch_noisy).item()

    #%%
    if STEPSIZEEXT:
        # Experiment: step size extension with different extension methods.
        start = time.time()
        def construct_extension(stepsizes, final_length, extend_type):
            """
            Generates extensions of stepsizes

            Args:
                stepsizes (Tensor): Initial stepsizes to extend
                final_length (int): Length of final stepsizes
                extend_type (string or float): method of extension, can be "max", "mean", "min", "final", "recip", or float quartile


            Returns:
                Tensor: Tensor of extended stepsizes, beginning with input stepsizes
            """
            num_to_extend = final_length - len(stepsizes)
            if num_to_extend < 0:
                raise Exception("final length less than number of steps")
            if extend_type == "max":
                to_cat = torch.ones(num_to_extend, device = stepsizes.device) * torch.max(stepsizes)
            elif extend_type == "mean":
                to_cat = torch.ones(num_to_extend, device = stepsizes.device) * torch.mean(stepsizes)
            elif extend_type == "min":
                to_cat = torch.ones(num_to_extend, device = stepsizes.device) * torch.min(stepsizes)
            elif extend_type == "final":
                to_cat = torch.ones(num_to_extend, device = stepsizes.device) * stepsizes[-1]
            elif extend_type == "recip":
                to_cat = torch.ones(num_to_extend, device = stepsizes.device)
                # first calculate c: taken as mean
                c = torch.mean(torch.arange(start=1, end = len(stepsizes)+ 1, device = stepsizes.device) * stepsizes)
                # then take c/k to concatenate
                to_cat = c/torch.arange(start = len(stepsizes)+ 1, end = final_length+1, device = stepsizes.device) 

            elif type(extend_type) is float: # quartile
                if extend_type < 0 or extend_type > 1:
                    ssmin = torch.min(stepsizes)
                    ssmax = torch.max(stepsizes)
                    to_cat = torch.ones(num_to_extend, device = stepsizes.device) * (extend_type*(ssmax-ssmin) + ssmin)
            else:
                raise Exception("Extension type not supported")
            
            extended_stepsizes = torch.cat((stepsizes, to_cat))
            return extended_stepsizes
        
        if PLOT:
            fig, ax = plt.subplots(figsize = (8,6), dpi = 150) # loss plot
            fig2, ax2 = plt.subplots(figsize = (8,6), dpi = 150) # fwdbwd plot

        # bookkeeping
        currminloss = torch.Tensor([init_err])
        currminfwdbwd = torch.minimum(torch.Tensor([init_closeness_mom]), torch.Tensor([init_closeness_nomom]))


        for extend_type in ["max", "mean", "min", "final", "recip"]:
            # construct extension
            new_ss_mom = construct_extension(adap_ss_mom, MAXITERS, extend_type)
            new_ss_nomom = construct_extension(adap_ss_nomom, MAXITERS, extend_type)
            print("mom", extend_type, new_ss_mom)
            print("mom", extend_type, new_ss_nomom)
            # initialize arrays
            loss_mom = [init_err]
            loss_nomom = [init_err]
            closeness_mom = [init_closeness_mom]
            closeness_nomom = [init_closeness_nomom]
            l2_nomom = [init_l2]
            l2_mom = [init_l2]
            # modify model
            icnn_couple_mom.stepsize = nn.Parameter(new_ss_mom)
            icnn_couple_nomom.stepsize = nn.Parameter(new_ss_nomom)

            # perform forward pass
            iterates_mom = icnn_couple_mom(fbp_batch_noisy, recon_err_grad)
            iterates_nomom = icnn_couple_nomom(fbp_batch_noisy1, recon_err_grad1)

            # check losses
            for i in iterates_mom:
                loss = recon_err(i).item()
                loss_mom.append(loss)
                closeness = icnn_couple_mom.fwdbwdloss(i).item()
                closeness_mom.append(closeness)
                l2_mom.append(torch.linalg.vector_norm(true_fwd - i).item())

            for i in iterates_nomom:
                loss = recon_err1(i).item()
                loss_nomom.append(loss)
                closeness = icnn_couple_nomom.fwdbwdloss(i).item()
                closeness_nomom.append(closeness)
                l2_nomom.append(torch.linalg.vector_norm(true_fwd1 - i).item())

            # book keeping
            currminloss = torch.minimum(currminloss, torch.min(torch.cat((torch.Tensor(loss_mom), torch.Tensor(loss_nomom)))))
            currminfwdbwd = torch.minimum(currminfwdbwd, torch.min(torch.cat((torch.Tensor(closeness_mom), torch.Tensor(closeness_nomom)))))
            
            # logging and plotting
            currtime = time.time()
            print(extend_type, "done, cumu time", currtime - start)
            # plot

            np.save(os.path.join("dat" + MODE, "stepsizeext", "loss_mom_" + extend_type), np.array(loss_mom))
            np.save(os.path.join("dat" + MODE, "stepsizeext", "loss_nomom_" + extend_type), np.array(loss_nomom))
            np.save(os.path.join("dat" + MODE, "stepsizeext", "fwdbwd_mom_" + extend_type), np.array(closeness_mom))
            np.save(os.path.join("dat" + MODE, "stepsizeext", "fwdbwd_nomom_" + extend_type), np.array(closeness_nomom))
            np.save(os.path.join("dat" + MODE, "stepsizeext", "l2_mom_" + extend_type), np.array(l2_mom))
            np.save(os.path.join("dat" + MODE, "stepsizeext", "l2_nomom_" + extend_type), np.array(l2_nomom))

            if PLOT:
                ax.plot(loss_mom, label = str(extend_type) + " mom")
                ax.plot(loss_nomom, label = str(extend_type) + " nomom")

                ax2.plot(closeness_mom, label = str(extend_type) + " mom")
                ax2.plot(closeness_nomom, label = str(extend_type) + " nomom")
        if PLOT:
            fig.suptitle("Loss")
            fig2.suptitle("Fwdbwd error")
            ax.set_ylim([currminloss.item()*0.95, init_err*5])
            ax2.set_ylim([currminfwdbwd.item()*0.95, init_closeness_mom*5])

            fig.legend()
            fig2.legend() 
            fig.savefig(os.path.join(figs_dir, "stepsize_ext_losses"))
            fig2.savefig(os.path.join(figs_dir, "stepsize_ext_fwdbwd"))
        end = time.time()
        print("adaptive stepsize extension done, elapsed time", end-start)

    #%%
    if CONSTSS:
        start = time.time()

        # TYPE 1: t_k = c
        for c in [0.1,0.05,0.01,0.005,0.001]:
            new_stepsizes = torch.ones(MAXITERS) * c
            icnn_couple_mom.stepsize = nn.Parameter(new_stepsizes.to(device0))
            icnn_couple_nomom.stepsize = nn.Parameter(new_stepsizes.to(device1))
            # init arrays
            loss_mom = [init_err]
            loss_nomom = [init_err]
            closeness_mom = [init_closeness_mom]
            closeness_nomom = [init_closeness_nomom]
            l2_nomom = [init_l2]
            l2_mom = [init_l2]

            # perform forward
            iterates_mom = icnn_couple_mom(fbp_batch_noisy, recon_err_grad)
            iterates_nomom = icnn_couple_nomom(fbp_batch_noisy1, recon_err_grad1)

            # check losses
            for i in iterates_mom:
                loss = recon_err(i).item()
                loss_mom.append(loss)
                closeness = icnn_couple_mom.fwdbwdloss(i).item()
                closeness_mom.append(closeness)
                l2_mom.append(torch.linalg.vector_norm(true_fwd - i).item())

            for i in iterates_nomom:
                loss = recon_err1(i).item()
                loss_nomom.append(loss)
                closeness = icnn_couple_nomom.fwdbwdloss(i).item()
                closeness_nomom.append(closeness)
                l2_nomom.append(torch.linalg.vector_norm(true_fwd1 - i).item())

            # save
            np.save(os.path.join("dat" + MODE, "constss", "loss_mom_" + str(c*1000)), np.array(loss_mom))
            np.save(os.path.join("dat" + MODE, "constss", "loss_nomom_" + str(c*1000)), np.array(loss_nomom))
            np.save(os.path.join("dat" + MODE, "constss", "fwdbwd_mom_" + str(c*1000)), np.array(closeness_mom))
            np.save(os.path.join("dat" + MODE, "constss", "fwdbwd_nomom_" + str(c*1000)), np.array(closeness_nomom))
            np.save(os.path.join("dat" + MODE, "constss", "l2_mom_" + str(c*1000)), np.array(l2_mom))
            np.save(os.path.join("dat" + MODE, "constss", "l2_nomom_" + str(c*1000)), np.array(l2_nomom))
        # TYPE 2: t_k = c/n
        for c in [0.1,0.08,0.05,0.02,0.01]:
            new_stepsizes = c/torch.arange(start=1, end=MAXITERS+1)
            icnn_couple_mom.stepsize = nn.Parameter(new_stepsizes.to(device0))
            icnn_couple_nomom.stepsize = nn.Parameter(new_stepsizes.to(device1))
            # init arrays
            loss_mom = [init_err]
            loss_nomom = [init_err]
            closeness_mom = [init_closeness_mom]
            closeness_nomom = [init_closeness_nomom]
            l2_nomom = [init_l2]
            l2_mom = [init_l2]

            # perform forward
            iterates_mom = icnn_couple_mom(fbp_batch_noisy, recon_err_grad)
            iterates_nomom = icnn_couple_nomom(fbp_batch_noisy1, recon_err_grad1)

            # check losses
            for i in iterates_mom:
                loss = recon_err(i).item()
                loss_mom.append(loss)
                closeness = icnn_couple_mom.fwdbwdloss(i).item()
                closeness_mom.append(closeness)
                l2_mom.append(torch.linalg.vector_norm(true_fwd - i).item())

            for i in iterates_nomom:
                loss = recon_err1(i).item()
                loss_nomom.append(loss)
                closeness = icnn_couple_nomom.fwdbwdloss(i).item()
                closeness_nomom.append(closeness)
                l2_nomom.append(torch.linalg.vector_norm(true_fwd1 - i).item())

            # save
            np.save(os.path.join("dat" + MODE, "recipss", "loss_mom_" + str(c*1000)), np.array(loss_mom))
            np.save(os.path.join("dat" + MODE, "recipss", "loss_nomom_" + str(c*1000)), np.array(loss_nomom))
            np.save(os.path.join("dat" + MODE, "recipss", "fwdbwd_mom_" + str(c*1000)), np.array(closeness_mom))
            np.save(os.path.join("dat" + MODE, "recipss", "fwdbwd_nomom_" + str(c*1000)), np.array(closeness_nomom))
            np.save(os.path.join("dat" + MODE, "recipss", "l2_mom_" + str(c*1000)), np.array(l2_mom))
            np.save(os.path.join("dat" + MODE, "recipss", "l2_nomom_" + str(c*1000)), np.array(l2_nomom))
        end = time.time()
        print("constant ss done, elapsed time", end-start)
    #%%
    if MAPTRANSFER:
        # reset
        icnn_couple_mom.stepsize = adap_ss_mom
        icnn_couple_nomom.stepsize = adap_ss_nomom
        loss_mom = [init_err]
        loss_nomom = [init_err]
        closeness_mom = [init_closeness_mom]
        closeness_nomom = [init_closeness_nomom]
        l2_nomom = [init_l2]
        l2_mom = [init_l2]


        iterates_mom = icnn_couple_mom(fbp_batch_noisy, recon_err_grad)
        iterates_nomom = icnn_couple_nomom(fbp_batch_noisy1, recon_err_grad1)

        # check losses
        for i in iterates_mom:
            loss = recon_err(i).item()
            loss_mom.append(loss)
            closeness = icnn_couple_mom.fwdbwdloss(i).item()
            closeness_mom.append(closeness)
            l2_mom.append(torch.linalg.vector_norm(true_fwd - i).item())

        for i in iterates_nomom:
            loss = recon_err1(i).item()
            loss_nomom.append(loss)
            closeness = icnn_couple_nomom.fwdbwdloss(i).item()
            closeness_nomom.append(closeness)
            l2_nomom.append(torch.linalg.vector_norm(true_fwd1 - i).item())

        np.save(os.path.join("dat" + MODE, "maptransfer", "standard", "loss_mom"), np.array(loss_mom))
        np.save(os.path.join("dat" + MODE, "maptransfer", "standard", "loss_nomom"), np.array(loss_nomom))
        np.save(os.path.join("dat" + MODE, "maptransfer", "standard", "fwdbwd_mom"), np.array(closeness_mom))
        np.save(os.path.join("dat" + MODE, "maptransfer", "standard", "fwdbwd_nomom"), np.array(closeness_nomom))
        np.save(os.path.join("dat" + MODE, "maptransfer", "standard", "l2_mom"), np.array(l2_mom))
        np.save(os.path.join("dat" + MODE, "maptransfer", "standard", "l2_nomom"), np.array(l2_nomom))
        
        # Redefine icnn with opposite map
        icnn_couple_mom = ICNNCoupleMomentum(device = device0, stepsize_init = 0.01, num_iters = 10, stepsize_clamp = (0.001,0.1))
        icnn_couple_mom.init_fwd(num_in_channels=1, num_filters=60, kernel_dim=3, num_layers=2, strong_convexity = 0.1, dense_size = 20)
        icnn_couple_mom.init_bwd(num_in_channels=1, num_filters=70, kernel_dim=3, num_layers=2, strong_convexity = 0.1, dense_size = 20)
        icnn_couple_mom.load_state_dict(torch.load(checkpoint_dir_nomom, map_location=device0))

        icnn_couple_nomom = ICNNCouple(device = device1, stepsize_init = 0.01, num_iters = 10, stepsize_clamp = (0.001,0.1))
        icnn_couple_nomom.init_fwd(num_in_channels=1, num_filters=60, kernel_dim=3, num_layers=2, strong_convexity = 0.1, dense_size = 20)
        icnn_couple_nomom.init_bwd(num_in_channels=1, num_filters=70, kernel_dim=3, num_layers=2, strong_convexity = 0.1, dense_size = 20)
        icnn_couple_nomom.load_state_dict(torch.load(checkpoint_dir_mom, map_location = device1))
        print("reinit with switched maps")
        icnn_couple_mom.eval()
        icnn_couple_nomom.eval()
        # simple 10 iter for now

        loss_mom = [init_err]
        loss_nomom = [init_err]
        closeness_mom = [init_closeness_nomom]
        closeness_nomom = [init_closeness_mom]
        l2_nomom = [init_l2]
        l2_mom = [init_l2]


        iterates_mom = icnn_couple_mom(fbp_batch_noisy, recon_err_grad)
        iterates_nomom = icnn_couple_nomom(fbp_batch_noisy1, recon_err_grad1)

        # check losses
        for i in iterates_mom:
            loss = recon_err(i).item()
            loss_mom.append(loss)
            closeness = icnn_couple_mom.fwdbwdloss(i).item()
            closeness_mom.append(closeness)
            l2_mom.append(torch.linalg.vector_norm(true_fwd - i).item())

        for i in iterates_nomom:
            loss = recon_err1(i).item()
            loss_nomom.append(loss)
            closeness = icnn_couple_nomom.fwdbwdloss(i).item()
            closeness_nomom.append(closeness)
            l2_nomom.append(torch.linalg.vector_norm(true_fwd1 - i).item())

        np.save(os.path.join("dat" + MODE, "maptransfer", "switch", "loss_mom"), np.array(loss_mom))
        np.save(os.path.join("dat" + MODE, "maptransfer", "switch", "loss_nomom"), np.array(loss_nomom))
        np.save(os.path.join("dat" + MODE, "maptransfer", "switch", "fwdbwd_mom"), np.array(closeness_mom))
        np.save(os.path.join("dat" + MODE, "maptransfer", "switch", "fwdbwd_nomom"), np.array(closeness_nomom))
        np.save(os.path.join("dat" + MODE, "maptransfer", "switch", "l2_mom"), np.array(l2_mom))
        np.save(os.path.join("dat" + MODE, "maptransfer", "switch", "l2_nomom"), np.array(l2_nomom))

        print("map transfer done")

    #%%
    if DOMAINCHANGE:
        print("domain change todo")
    
    '''
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
    '''