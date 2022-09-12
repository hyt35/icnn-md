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
from torchvision.utils import make_grid, save_image
from odl import uniform_discr
from dival.util.odl_utility import ResizeOperator
#%% EXPERIMENT PARAMETERS
IMPL = 'astra_cuda'
# MODE = "ELLIPSE"
MODE = "LODOPAB"
EXPTYPE = "MAN" # experiment type 
MAXITERS = 1000 # maximum number of optim iter for experiments
PLOT = False # use the other script to plot.
LONGTIME = True
#%% EXPERIMENT FLAGS
if EXPTYPE == "ALL":
    GRABNEW = True
    COMPUTETRUEMIN = True
    STEPSIZEEXT = True
    CONSTSS = True
    MAPTRANSFER = True
    DOMAINCHANGE = True
    ALTTRANSFORM = True
    DECONV = True
    
else: # manual
    GRABNEW = False
    COMPUTETRUEMIN = False
    STEPSIZEEXT = False
    CONSTSS = False
    MAPTRANSFER = False
    DOMAINCHANGE = False
    ALTTRANSFORM = True
    DECONV = False

#%% INITIALIZATION
device0 = 'cuda:0'
device1 = 'cuda:1'

workdir_mom = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/fbp_mom"
workdir_nomom = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/fbp_nomom"

checkpoint_dir_mom = os.path.join(workdir_mom, "checkpoints", "20")
checkpoint_dir_nomom = os.path.join(workdir_nomom, "checkpoints", "20")

if MODE == "ELLIPSE":
    figs_dir = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/figs/ellipse"
    datpath = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/datELLIPSE"
else:
    figs_dir = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/figs/lodopab"
    datpath = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/datLODOPAB"

if LONGTIME:
    figs_dir = figs_dir + "_longtime"
    datpath = datpath+"long"
args=parse_import.parse_commandline_args()

#%%
# create subfolder
tf.io.gfile.makedirs(datpath)
tf.io.gfile.makedirs(os.path.join(datpath, "stepsizeext"))
tf.io.gfile.makedirs(os.path.join(datpath, "constss"))
tf.io.gfile.makedirs(os.path.join(datpath, "recipss"))
tf.io.gfile.makedirs(os.path.join(datpath, "maptransfer"))
tf.io.gfile.makedirs(os.path.join(datpath, "maptransfer", "standard"))
tf.io.gfile.makedirs(os.path.join(datpath, "maptransfer", "switch"))
tf.io.gfile.makedirs(os.path.join(datpath, "alttransform"))
tf.io.gfile.makedirs(os.path.join(datpath, "deconv"))
#%%
# Load models

icnn_couple_mom = ICNNCoupleMomentum(device = device0, stepsize_init = 0.01, num_iters = 10, stepsize_clamp = (0.001,0.1))
icnn_couple_mom.init_fwd(num_in_channels=1, num_filters=60, kernel_dim=3, num_layers=2, strong_convexity = 0.1, dense_size = 20)
icnn_couple_mom.init_bwd(num_in_channels=1, num_filters=70, kernel_dim=3, num_layers=2, strong_convexity = 0.1, dense_size = 20)
icnn_couple_mom.load_state_dict(torch.load(checkpoint_dir_mom, map_location=device0))
icnn_couple_mom.device = device0

icnn_couple_nomom = ICNNCouple(device = device1, stepsize_init = 0.01, num_iters = 10, stepsize_clamp = (0.001,0.1))
icnn_couple_nomom.init_fwd(num_in_channels=1, num_filters=60, kernel_dim=3, num_layers=2, strong_convexity = 0.1, dense_size = 20)
icnn_couple_nomom.init_bwd(num_in_channels=1, num_filters=70, kernel_dim=3, num_layers=2, strong_convexity = 0.1, dense_size = 20)
icnn_couple_nomom.load_state_dict(torch.load(checkpoint_dir_nomom, map_location = device1))
icnn_couple_nomom.device = device1

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
        torch.save(gt, os.path.join(datpath, MODE+"gt.pt"))
        torch.save(fbp_batch_noisy, os.path.join(datpath, MODE+"batch.pt"))
        fbp_batch_noisy1 = fbp_batch_noisy.clone().to(device1)
    else:
        gt = torch.load(os.path.join(datpath, MODE+"gt.pt")).to(device0)
        fbp_batch_noisy = torch.load(os.path.join(datpath, MODE+"batch.pt")).to(device0)
        fbp_batch_noisy1 = fbp_batch_noisy.clone().to(device1)

        with tf.io.gfile.GFile(os.path.join(figs_dir, "gt.png"), "wb") as fout:
            save_image(gt.detach().cpu(), fout, nrow = 5)
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
        gdloss = [init_err]
        fwd = fbp_batch_noisy.clone().detach().requires_grad_(True)
        for i in range(2000):
            fwd = fwd.detach().requires_grad_(True)
            fwdGrad = recon_err_grad(fwd)
            fwd = fwd - 5e-4*fwdGrad
            gdloss.append(recon_err(fwd).item())
        for i in range(5000):
            fwd = fwd.detach().requires_grad_(True)
            fwdGrad = recon_err_grad(fwd)
            fwd = fwd - 5e-4*fwdGrad
            gdloss.append(recon_err(fwd).item())
        # plot
        true_min = recon_err(fwd).item()
        np.save(os.path.join(datpath, "true_gd_progression"), gdloss)
        np.save(os.path.join(datpath, "true_min_val"), true_min)
        np.save(os.path.join(datpath, "true_min_arr"), fwd.detach().cpu().numpy())
        print("compute true min", true_min)

        nesterovloss = [init_err]
        lam = 0
        currentstep = 1
        yk = fbp_batch_noisy.clone().detach().requires_grad_(True)
        xk = yk
        for i in range(1000):
            yk = yk.detach().requires_grad_(True)
            xklast = xk
            xk = yk - 5e-4*recon_err_grad(yk)
            yk = xk + i/(i+3)*(xk-xklast)

            nesterovloss.append(recon_err(yk).item())

        np.save(os.path.join(datpath, "nesterov_progression"), nesterovloss)
    
    true_min = np.load(os.path.join(datpath, "true_min_val.npy"))
    true_fwd = torch.from_numpy(np.load(os.path.join(datpath, "true_min_arr.npy"))).to(device0)
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

            fwd_mom = iterates_mom[-1]
            fwd_nomom = iterates_nomom[-1]
            # book keeping
            currminloss = torch.minimum(currminloss, torch.min(torch.cat((torch.Tensor(loss_mom), torch.Tensor(loss_nomom)))))
            currminfwdbwd = torch.minimum(currminfwdbwd, torch.min(torch.cat((torch.Tensor(closeness_mom), torch.Tensor(closeness_nomom)))))
            
            # logging and plotting
            currtime = time.time()
            print(extend_type, "done, cumu time", currtime - start)
            # plot

            np.save(os.path.join(datpath, "stepsizeext", "loss_mom_" + extend_type), np.array(loss_mom))
            np.save(os.path.join(datpath, "stepsizeext", "loss_nomom_" + extend_type), np.array(loss_nomom))
            np.save(os.path.join(datpath, "stepsizeext", "fwdbwd_mom_" + extend_type), np.array(closeness_mom))
            np.save(os.path.join(datpath, "stepsizeext", "fwdbwd_nomom_" + extend_type), np.array(closeness_nomom))
            np.save(os.path.join(datpath, "stepsizeext", "l2_mom_" + extend_type), np.array(l2_mom))
            np.save(os.path.join(datpath, "stepsizeext", "l2_nomom_" + extend_type), np.array(l2_nomom))

            if PLOT:
                ax.plot(loss_mom, label = str(extend_type) + " mom")
                ax.plot(loss_nomom, label = str(extend_type) + " nomom")

                ax2.plot(closeness_mom, label = str(extend_type) + " mom")
                ax2.plot(closeness_nomom, label = str(extend_type) + " nomom")

                with tf.io.gfile.GFile(os.path.join(figs_dir, "fwd_mom"+extend_type+".png"), "wb") as fout:
                    save_image(fwd_mom.detach().cpu(), fout, nrow = 5)
                with tf.io.gfile.GFile(os.path.join(figs_dir,  "fwd_nomom"+extend_type+".png"), "wb") as fout:
                    save_image(fwd_nomom.detach().cpu(), fout, nrow = 5)
        # if PLOT:
        #     fig.suptitle("Loss")
        #     fig2.suptitle("Fwdbwd error")
        #     ax.set_ylim([currminloss.item()*0.95, init_err*5])
        #     ax2.set_ylim([currminfwdbwd.item()*0.95, init_closeness_mom*5])

        #     fig.legend()
        #     fig2.legend() 
        #     fig.savefig(os.path.join(figs_dir, "stepsize_ext_losses"))
        #     fig2.savefig(os.path.join(figs_dir, "stepsize_ext_fwdbwd"))
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
            np.save(os.path.join(datpath, "constss", "loss_mom_" + str(c*1000)), np.array(loss_mom))
            np.save(os.path.join(datpath, "constss", "loss_nomom_" + str(c*1000)), np.array(loss_nomom))
            np.save(os.path.join(datpath, "constss", "fwdbwd_mom_" + str(c*1000)), np.array(closeness_mom))
            np.save(os.path.join(datpath, "constss", "fwdbwd_nomom_" + str(c*1000)), np.array(closeness_nomom))
            np.save(os.path.join(datpath, "constss", "l2_mom_" + str(c*1000)), np.array(l2_mom))
            np.save(os.path.join(datpath, "constss", "l2_nomom_" + str(c*1000)), np.array(l2_nomom))
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
            np.save(os.path.join(datpath, "recipss", "loss_mom_" + str(c*1000)), np.array(loss_mom))
            np.save(os.path.join(datpath, "recipss", "loss_nomom_" + str(c*1000)), np.array(loss_nomom))
            np.save(os.path.join(datpath, "recipss", "fwdbwd_mom_" + str(c*1000)), np.array(closeness_mom))
            np.save(os.path.join(datpath, "recipss", "fwdbwd_nomom_" + str(c*1000)), np.array(closeness_nomom))
            np.save(os.path.join(datpath, "recipss", "l2_mom_" + str(c*1000)), np.array(l2_mom))
            np.save(os.path.join(datpath, "recipss", "l2_nomom_" + str(c*1000)), np.array(l2_nomom))
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

        np.save(os.path.join(datpath, "maptransfer", "standard", "loss_mom"), np.array(loss_mom))
        np.save(os.path.join(datpath, "maptransfer", "standard", "loss_nomom"), np.array(loss_nomom))
        np.save(os.path.join(datpath, "maptransfer", "standard", "fwdbwd_mom"), np.array(closeness_mom))
        np.save(os.path.join(datpath, "maptransfer", "standard", "fwdbwd_nomom"), np.array(closeness_nomom))
        np.save(os.path.join(datpath, "maptransfer", "standard", "l2_mom"), np.array(l2_mom))
        np.save(os.path.join(datpath, "maptransfer", "standard", "l2_nomom"), np.array(l2_nomom))
        
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

        np.save(os.path.join(datpath, "maptransfer", "switch", "loss_mom"), np.array(loss_mom))
        np.save(os.path.join(datpath, "maptransfer", "switch", "loss_nomom"), np.array(loss_nomom))
        np.save(os.path.join(datpath, "maptransfer", "switch", "fwdbwd_mom"), np.array(closeness_mom))
        np.save(os.path.join(datpath, "maptransfer", "switch", "fwdbwd_nomom"), np.array(closeness_nomom))
        np.save(os.path.join(datpath, "maptransfer", "switch", "l2_mom"), np.array(l2_mom))
        np.save(os.path.join(datpath, "maptransfer", "switch", "l2_nomom"), np.array(l2_nomom))

        print("map transfer done")

    #%%
    if DOMAINCHANGE:
        print("domain change todo")
    if ALTTRANSFORM:
        # Define new transform

        if MODE == "ELLIPSE":
            def fbp_postprocess(fbp): # no correction needed for ellipses dataset
                foo = fbp.view(fbp.shape[0], -1)
                min_ = torch.min(foo, dim = 1)[0]
                max_ = torch.max(foo, dim = 1)[0]
                lowquantile = torch.quantile(foo, 0.05, dim=1)[:,None,None,None]
                min__ = min_[:,None,None,None]
                max__ = max_[:,None,None,None]
                fbp = (fbp-min__)/(max__-min__)
                # fbp = (fbp - min__)/(max__-lowquantile)

                new = torch.zeros_like(fbp)
                mask = fbp > lowquantile
                new[mask] = fbp[mask]
                fbp = new
                return fbp
        else:
            def fbp_postprocess(fbp): # correction for lodopab fbp inversion issue
                foo = fbp.view(fbp.shape[0], -1)
                min_ = torch.min(foo, dim = 1)[0]
                max_ = torch.max(foo, dim = 1)[0]
                lowquantile = torch.quantile(foo, 0.05, dim=1)[:,None,None,None]
                min__ = min_[:,None,None,None]
                max__ = max_[:,None,None,None]
                fbp = (fbp-min__)/(max__-min__)
                # fbp = (fbp - min__)/(max__-lowquantile)

                new = torch.zeros_like(fbp)
                mask = fbp > lowquantile
                new[mask] = fbp[mask]
                fbp = new

                # fbp.clamp_(0)
                
                return fbp

        try:
            fbp_batch_noisy = torch.load(os.path.join(datpath, MODE+"batch_altrecon.pt"))
            print("succesful grab")
        except FileNotFoundError:
            print("create new recon")
            min_pt = [-128/2, -128/2]
            max_pt = [128/2, 128/2]
            reco_space = uniform_discr(min_pt, max_pt, (128,128), dtype=np.float32)

            NUM_ANGLES = 60
            DET_SHAPE = 400
            # image shape for simulation
            IM_SHAPE = (400,400)  # images will be scaled up from (128, 128)

            space = odl.uniform_discr(min_pt=reco_space.min_pt,
                                    max_pt=reco_space.max_pt,
                                    shape=IM_SHAPE, dtype=np.float32) # upscale
            reco_geometry = odl.tomo.parallel_beam_geometry(
                reco_space, num_angles=NUM_ANGLES)
            geometry = odl.tomo.parallel_beam_geometry(
                space, num_angles=NUM_ANGLES,
                det_shape=reco_geometry.detector.shape)

            resize_up = torchvision.transforms.Resize((400,400))
            resize_down = torchvision.transforms.Resize((128,128))
            
            impl = IMPL
            space_range = 64 # imsize/2

            geometry = odl.tomo.geometry.conebeam.cone_beam_geometry(space, src_radius=1.6*space_range, \
                                                                    det_radius=1.6*space_range, num_angles=NUM_ANGLES, det_shape=DET_SHAPE)
            
        
            fwd_op_odl = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
            fbp_op_odl = odl.tomo.fbp_op(fwd_op_odl)
            
            fwd_op = torch_wrapper.OperatorModule(fwd_op_odl).to(device0)
            fbp_op = torch_wrapper.OperatorModule(fbp_op_odl).to(device0)
            
            #print('op_norm = %.4f'%(1.1 * odl.power_method_opnorm(fwd_op_odl)))
            
            gt_clone = resize_up(gt.clone().detach().to(device0))
            sinogram = fwd_op(gt_clone)
            sinogram_noisy = sinogram + noise_level*torch.randn(sinogram.size()).to(device0)
            
            fbp_batch_noisy = resize_down(fbp_op(sinogram_noisy))

            torch.save(fbp_batch_noisy, os.path.join(datpath, MODE+"batch_altrecon.pt"))
        
        fbp_batch_noisy = fbp_postprocess(fbp_batch_noisy).to(device0)
        fbp_batch_noisy1 = fbp_batch_noisy.clone().detach().to(device1)
        #print("min", torch.min(fbp_batch_noisy), "max", torch.max(fbp_batch_noisy))
        with tf.io.gfile.GFile(os.path.join(figs_dir, "altbatch.png"), "wb") as fout:
                    save_image(fbp_batch_noisy.detach().cpu(), fout, nrow = 5)

        
        # redefine loss functions with new fbp batch 
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
        
        MAXITERS = 200
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

            fwd_mom = iterates_mom[-1]
            fwd_nomom = iterates_nomom[-1]

            with tf.io.gfile.GFile(os.path.join(figs_dir, "alttransform_mom"+str(c)+".png"), "wb") as fout:
                save_image(fwd_mom.detach().cpu(), fout, nrow = 5)
            with tf.io.gfile.GFile(os.path.join(figs_dir,  "alttransform_nomom"+str(c)+".png"), "wb") as fout:
                save_image(fwd_nomom.detach().cpu(), fout, nrow = 5)
            # save
            np.save(os.path.join(datpath, "alttransform", "loss_mom_" + str(c*1000)), np.array(loss_mom))
            np.save(os.path.join(datpath, "alttransform", "loss_nomom_" + str(c*1000)), np.array(loss_nomom))
            np.save(os.path.join(datpath, "alttransform", "fwdbwd_mom_" + str(c*1000)), np.array(closeness_mom))
            np.save(os.path.join(datpath, "alttransform", "fwdbwd_nomom_" + str(c*1000)), np.array(closeness_nomom))
            np.save(os.path.join(datpath, "alttransform", "l2_mom_" + str(c*1000)), np.array(l2_mom))
            np.save(os.path.join(datpath, "alttransform", "l2_nomom_" + str(c*1000)), np.array(l2_nomom))
        end = time.time()
        print("alt ray transform")
        

        print("alterative ray transform done")


    if DECONV:
        gt = torch.load(os.path.join(datpath, MODE+"gt.pt")).to(device0)

        # apply blur and gaussian
        blur_op = torchvision.transforms.GaussianBlur(kernel_size = 5, sigma=2) 
        batch_noisy = blur_op(gt) + 0.05*torch.randn_like(gt)
        batch_noisy1 = batch_noisy.clone().detach().to(device1)
        with tf.io.gfile.GFile(os.path.join(figs_dir, "deconv_batch.png"), "wb") as fout:
                save_image(batch_noisy.detach().cpu(), fout, nrow = 5)

        # define new fwd ops, with blur kernel
        def recon_err(img):
            tv_h = torch.abs(img[:,:,1:,:]-img[:,:,:-1,:]).sum()
            tv_w = torch.abs(img[:,:,:,1:]-img[:,:,:,:-1]).sum()
            tv = (tv_h+tv_w)
            fidelity = torch.pow((blur_op(img)-batch_noisy),2).sum()
            return (fidelity + reg_param*tv)/2

        def recon_err_grad(img):
            return autograd.grad(recon_err(img), img)[0]
        def recon_err1(img):
            tv_h = torch.abs(img[:,:,1:,:]-img[:,:,:-1,:]).sum()
            tv_w = torch.abs(img[:,:,:,1:]-img[:,:,:,:-1]).sum()
            tv = (tv_h+tv_w)
            fidelity = torch.pow((blur_op(img)-batch_noisy1),2).sum()
            return (fidelity + reg_param*tv)/2
        
        def recon_err_grad1(img):
            return autograd.grad(recon_err1(img), img)[0]

        # perform model forward
        MAXITERS = 200
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
            iterates_mom = icnn_couple_mom(batch_noisy, recon_err_grad)
            iterates_nomom = icnn_couple_nomom(batch_noisy1, recon_err_grad1)

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

            fwd_mom = iterates_mom[-1]
            fwd_nomom = iterates_nomom[-1]

            with tf.io.gfile.GFile(os.path.join(figs_dir, "deconv_transfer_mom"+str(c)+".png"), "wb") as fout:
                save_image(fwd_mom.detach().cpu(), fout, nrow = 5)
            with tf.io.gfile.GFile(os.path.join(figs_dir,  "deconv_transfer_nomom"+str(c)+".png"), "wb") as fout:
                save_image(fwd_nomom.detach().cpu(), fout, nrow = 5)
            # save
            np.save(os.path.join(datpath, "deconv", "loss_mom_" + str(c*1000)), np.array(loss_mom))
            np.save(os.path.join(datpath, "deconv", "loss_nomom_" + str(c*1000)), np.array(loss_nomom))
            np.save(os.path.join(datpath, "deconv", "fwdbwd_mom_" + str(c*1000)), np.array(closeness_mom))
            np.save(os.path.join(datpath, "deconv", "fwdbwd_nomom_" + str(c*1000)), np.array(closeness_nomom))
            np.save(os.path.join(datpath, "deconv", "l2_mom_" + str(c*1000)), np.array(l2_mom))
            np.save(os.path.join(datpath, "deconv", "l2_nomom_" + str(c*1000)), np.array(l2_nomom))
        print("deconv fwd op done")
