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

#%% EXPERIMENT PARAMETERS
IMPL = 'astra_cuda'
# MODE = "ELLIPSE"
MODE = "LODOPAB"
EXPTYPE = "MAN" # experiment type 
LONGTIME = True
MOMSPLIT = False
PDF = True
PNG = True
eps=1e-6
#%% EXPERIMENT FLAGS
if EXPTYPE == "ALL":
    STEPSIZEEXT = True
    CONSTSS = True
    MAPTRANSFER = True
    DOMAINCHANGE = True
    ALTTRANSFORM = True
    DENOISE = True
    CROSS = True   
else: # manual
    STEPSIZEEXT = True
    CONSTSS = True
    MAPTRANSFER = False
    DOMAINCHANGE = False
    ALTTRANSFORM = True
    DENOISE = True
    CROSS = True
#%% INITIALIZATION
device0 = 'cuda:1'
device1 = 'cuda:2'

workdir_mom = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/fbp_mom"
workdir_nomom = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/fbp_nomom"

checkpoint_dir_mom = os.path.join(workdir_mom, "checkpoints", "20")
checkpoint_dir_nomom = os.path.join(workdir_nomom, "checkpoints", "20")

if MODE == "ELLIPSE":
    figs_dir = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/figs_deconv/ellipse"
    datpath = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/datELLIPSE_deconv"
else:
    figs_dir = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/figs_deconv/lodopab"
    datpath = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/datLODOPAB_deconv"

if LONGTIME:
    figs_dir = figs_dir + "_longtime"
    datpath = datpath+"long"
#%%
tf.io.gfile.makedirs(figs_dir)
#%%
args=parse_import.parse_commandline_args()
true_min = np.load(os.path.join(datpath, "true_min_val.npy"))
gdloss = np.load(os.path.join(datpath, "true_gd_progression.npy"))
#gdloss_recip = np.load(os.path.join(datpath, "recip_gd_progression.npy"))
nesterovloss = np.load(os.path.join(datpath, "nesterov_progression.npy"))
if STEPSIZEEXT:
    # Experiment: step size extension with different extension methods.
    start = time.time()

    fig, ax = plt.subplots(figsize = (6.4,4.8), dpi = 150) # loss plot
    fig2, ax2 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # fwdbwd plot
    fig3, ax3 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # log-loss plot 
    ax3.set_yscale('log')
    if LONGTIME:
        ax3.set_xscale('log')
        ax3.set_xlim(right = 10**3)
    ax3.plot(gdloss[:10**3]- true_min + eps, color='k', label = 'gd')
    ax3.plot(nesterovloss[:10**3]- true_min + eps, color='maroon', label = 'nesterov')
    fig4, ax4 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # l2 distance to minimum
    # bookkeeping: find minimum loss and fwdbwd
    # currminloss = None
    currminfwdbwd = None
    init_err = None
    init_closeness_mom = None
    init_l2 = None
    
    ax.axhline(true_min, label = "true min", linestyle = '--')
    #ax3.axhline(true_min, label = "true min", linestyle = '--')
    print(true_min)

    min_attain = np.amin(np.concatenate((gdloss[:10**3], nesterovloss[:10**3])))

    # loop 2: now plot.
    for extend_type in ["max", "mean", "min", "final", "recip"]:
        loss_mom = np.load(os.path.join(datpath, "stepsizeext", "loss_mom_" + extend_type + ".npy"))
        loss_nomom = np.load(os.path.join(datpath, "stepsizeext", "loss_nomom_" + extend_type + ".npy"))
        closeness_mom = np.load(os.path.join(datpath, "stepsizeext", "fwdbwd_mom_" + extend_type + ".npy"))
        closeness_nomom = np.load(os.path.join(datpath, "stepsizeext", "fwdbwd_nomom_" + extend_type + ".npy"))
        l2_mom = np.load(os.path.join(datpath, "stepsizeext", "l2_mom_" + extend_type + ".npy"))
        l2_nomom = np.load(os.path.join(datpath, "stepsizeext", "l2_nomom_" + extend_type + ".npy"))

        min_attain = np.amin(np.concatenate(([min_attain], loss_mom, loss_nomom)))
        if currminfwdbwd is None:
            currminfwdbwd = np.amin(np.concatenate((closeness_mom, closeness_nomom)))
        else:
            currminfwdbwd = np.amin((currminfwdbwd, np.amin(np.concatenate((closeness_mom, closeness_nomom)))))
        if init_err is None:
            init_err = loss_mom[0]
        if init_closeness_mom is None:
            init_closeness_mom = closeness_mom[0]
        if init_l2 is None:
            init_l2 = l2_mom[0]

        ax.plot(loss_mom, label = str(extend_type) + " mom", marker = 'o')
        ax.plot(loss_nomom, label = str(extend_type) + " nomom", marker = 'x')

        ax2.plot(closeness_mom, label = str(extend_type) + " mom", marker = 'o')
        ax2.plot(closeness_nomom, label = str(extend_type) + " nomom", marker = 'x')

        ax3.plot(loss_mom - true_min + eps, label = str(extend_type) + " mom", marker = 'o')
        ax3.plot(loss_nomom - true_min + eps, label = str(extend_type) + " nomom", marker = 'x')

        ax4.plot(l2_mom, label = str(extend_type) + " mom", marker = 'o')
        ax4.plot(l2_nomom, label = str(extend_type) + " nomom", marker = 'x')


    fig.suptitle("Loss")
    fig2.suptitle("Fwdbwd error")
    fig3.suptitle("Loss minus minimum recon")
    fig4.suptitle("L2 distance to true min")

    ax.set_ylim([true_min.item()*0.95, init_err*1.5])
    ax2.set_ylim([currminfwdbwd.item()*0.95, init_closeness_mom*1.5])
    ax3.set_ylim([(min_attain-true_min)*0.9, (init_err-true_min)*2.5])
    ax4.set_ylim([0,init_l2*1.5])

    fig.legend()
    fig2.legend() 
    fig3.legend()
    fig4.legend()
    if PDF:
        fig.savefig(os.path.join(figs_dir, "stepsize_ext_losses.pdf"))
        fig2.savefig(os.path.join(figs_dir, "stepsize_ext_fwdbwd.pdf"))
        fig3.savefig(os.path.join(figs_dir, "stepsize_ext_loglosses.pdf"))
        fig4.savefig(os.path.join(figs_dir, "stepsize_ext_l2totrue.pdf"))
    if PNG:
        fig.savefig(os.path.join(figs_dir, "stepsize_ext_losses"))
        fig2.savefig(os.path.join(figs_dir, "stepsize_ext_fwdbwd"))
        fig3.savefig(os.path.join(figs_dir, "stepsize_ext_loglosses"))
        fig4.savefig(os.path.join(figs_dir, "stepsize_ext_l2totrue"))
    end = time.time()
    print("adaptive stepsize extension done, elapsed time", end-start)

#%%
if CONSTSS:
    start = time.time()
    # type 1
    fig, ax = plt.subplots(figsize = (6.4,4.8), dpi = 150) # loss plot
    fig2, ax2 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # fwdbwd plot
    fig3, ax3 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # log-loss plot 
    ax3.set_yscale('log')
    if LONGTIME:
        ax3.set_xscale('log')
        ax3.set_xlim(right = 10**3)
    ax3.plot(gdloss[:10**3]- true_min + eps, color='k', label = 'gd')
    ax3.plot(nesterovloss[:10**3]- true_min + eps, color='maroon', label = 'nesterov')
    fig4, ax4 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # l2 distance to minimum

    ax.axhline(true_min, label = "true min", linestyle = '--')
    min_attain = np.amin(np.concatenate((gdloss[:10**3], nesterovloss[:10**3])))

    init_err = None
    init_closeness_mom = None
    init_l2 = None
    # TYPE 1: t_k = c
    for c in [0.1,0.05,0.01,0.005,0.001]:
        # load
        loss_mom = np.load(os.path.join(datpath, "constss", "loss_mom_" + str(c*1000) + ".npy"))
        loss_nomom = np.load(os.path.join(datpath, "constss", "loss_nomom_" + str(c*1000) + ".npy"))
        closeness_mom = np.load(os.path.join(datpath, "constss", "fwdbwd_mom_" + str(c*1000) + ".npy"))
        closeness_nomom = np.load(os.path.join(datpath, "constss", "fwdbwd_nomom_" + str(c*1000) + ".npy"))
        l2_mom = np.load(os.path.join(datpath, "constss", "l2_mom_" + str(c*1000) + ".npy"))
        l2_nomom = np.load(os.path.join(datpath, "constss", "l2_nomom_" + str(c*1000) + ".npy"))
        min_attain = np.amin(np.concatenate(([min_attain], loss_mom, loss_nomom)))
        if init_err is None:
            init_err = loss_mom[0]
        if init_closeness_mom is None:
            init_closeness_mom = closeness_mom[0]
        if init_l2 is None:
            init_l2 = l2_mom[0]


        ax.plot(loss_mom, label = str(c) + " mom", marker = 'o')
        ax.plot(loss_nomom, label = str(c) + " nomom", marker = 'x')

        ax2.plot(closeness_mom, label = str(c) + " mom", marker = 'o')
        ax2.plot(closeness_nomom, label = str(c) + " nomom", marker = 'x')

        ax3.plot(loss_mom - true_min + eps, label = str(c) + " mom", marker = 'o')
        ax3.plot(loss_nomom - true_min + eps, label = str(c) + " nomom", marker = 'x')

        ax4.plot(l2_mom, label = str(c) + " mom", marker = 'o')
        ax4.plot(l2_nomom, label = str(c) + " nomom", marker = 'x')

    fig.suptitle("Loss")
    fig2.suptitle("Fwdbwd error")
    fig3.suptitle("Loss minus minimum recon")
    fig4.suptitle("L2 distance to true min")

    ax.set_ylim([true_min.item()*0.95, init_err*1.5])
    ax2.set_ylim([currminfwdbwd.item()*0.95, init_closeness_mom*1.5])
    ax3.set_ylim([(min_attain-true_min)*0.9, (init_err-true_min)*2.5])
    ax4.set_ylim([0,init_l2*1.5])

    
    fig.legend()
    fig2.legend() 
    fig3.legend()
    fig4.legend()
    if PDF:
        fig.savefig(os.path.join(figs_dir, "constss_losses.pdf"))
        fig2.savefig(os.path.join(figs_dir, "constss_fwdbwd.pdf"))
        fig3.savefig(os.path.join(figs_dir, "constss_loglosses.pdf"))
        fig4.savefig(os.path.join(figs_dir, "constss_l2totrue.pdf"))
    if PNG:
        fig.savefig(os.path.join(figs_dir, "constss_losses"))
        fig2.savefig(os.path.join(figs_dir, "constss_fwdbwd"))
        fig3.savefig(os.path.join(figs_dir, "constss_loglosses"))
        fig4.savefig(os.path.join(figs_dir, "constss_l2totrue"))
    # type 2
    fig, ax = plt.subplots(figsize = (6.4,4.8), dpi = 150) # loss plot
    fig2, ax2 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # fwdbwd plot
    fig3, ax3 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # log-loss plot 
    ax3.set_yscale('log')
    if LONGTIME:
        ax3.set_xscale('log')
        ax3.set_xlim(right = 10**3)
    ax3.plot(gdloss[:10**3]- true_min + eps, color='k', label = 'gd')
    ax3.plot(nesterovloss[:10**3]- true_min + eps, color='maroon', label = 'nesterov')
    fig4, ax4 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # l2 distance to minimum

    ax.axhline(true_min, label = "true min", linestyle = '--')
    min_attain = np.amin(np.concatenate((gdloss[:10**3], nesterovloss[:10**3])))

    # TYPE 2: t_k = c/k
    for c in [0.1,0.08,0.05,0.02,0.01]:
        # load
        loss_mom = np.load(os.path.join(datpath, "recipss", "loss_mom_" + str(c*1000) + ".npy"))
        loss_nomom = np.load(os.path.join(datpath, "recipss", "loss_nomom_" + str(c*1000) + ".npy"))
        closeness_mom = np.load(os.path.join(datpath, "recipss", "fwdbwd_mom_" + str(c*1000) + ".npy"))
        closeness_nomom = np.load(os.path.join(datpath, "recipss", "fwdbwd_nomom_" + str(c*1000) + ".npy"))
        l2_mom = np.load(os.path.join(datpath, "recipss", "l2_mom_" + str(c*1000) + ".npy"))
        l2_nomom = np.load(os.path.join(datpath, "recipss", "l2_nomom_" + str(c*1000) + ".npy"))
        min_attain = np.amin(np.concatenate(([min_attain], loss_mom, loss_nomom)))        
        ax.plot(loss_mom, label = str(c) + " mom", marker = 'o')
        ax.plot(loss_nomom, label = str(c) + " nomom", marker = 'x')

        ax2.plot(closeness_mom, label = str(c) + " mom", marker = 'o')
        ax2.plot(closeness_nomom, label = str(c) + " nomom", marker = 'x')

        ax3.plot(loss_mom - true_min + eps, label = str(c) + " mom", marker = 'o')
        ax3.plot(loss_nomom - true_min + eps, label = str(c) + " nomom", marker = 'x')

        ax4.plot(l2_mom, label = str(c) + " mom", marker = 'o')
        ax4.plot(l2_nomom, label = str(c) + " nomom", marker = 'x')

    fig.suptitle("Loss")
    fig2.suptitle("Fwdbwd error")
    fig3.suptitle("Loss minus minimum recon")
    fig4.suptitle("L2 distance to true min")

    ax.set_ylim([true_min.item()*0.95, init_err*1.5])
    ax2.set_ylim([currminfwdbwd.item()*0.95, init_closeness_mom*1.5])
    ax3.set_ylim([(min_attain-true_min)*0.9, (init_err-true_min)*2.5])
    ax4.set_ylim([0,init_l2*1.5])

    fig.legend()
    fig2.legend() 
    fig3.legend()
    fig4.legend()
    if PDF:
        fig.savefig(os.path.join(figs_dir, "recipss_losses.pdf"))
        fig2.savefig(os.path.join(figs_dir, "recipss_fwdbwd.pdf"))
        fig3.savefig(os.path.join(figs_dir, "recipss_loglosses.pdf"))
        fig4.savefig(os.path.join(figs_dir, "recipss_l2totrue.pdf"))
    if PNG:
        fig.savefig(os.path.join(figs_dir, "recipss_losses"))
        fig2.savefig(os.path.join(figs_dir, "recipss_fwdbwd"))
        fig3.savefig(os.path.join(figs_dir, "recipss_loglosses"))
        fig4.savefig(os.path.join(figs_dir, "recipss_l2totrue"))

    # type 3
    fig, ax = plt.subplots(figsize = (6.4,4.8), dpi = 150) # loss plot
    fig2, ax2 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # fwdbwd plot
    fig3, ax3 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # log-loss plot 
    ax3.set_yscale('log')
    if LONGTIME:
        ax3.set_xscale('log')
        ax3.set_xlim(right = 10**3)
    ax3.plot(gdloss[:10**3]- true_min + eps, color='k', label = 'gd')
    ax3.plot(nesterovloss[:10**3]- true_min + eps, color='maroon', label = 'nesterov')
    fig4, ax4 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # l2 distance to minimum

    ax.axhline(true_min, label = "true min", linestyle = '--')
    min_attain = np.amin(np.concatenate((gdloss[:10**3], nesterovloss[:10**3])))

    # TYPE 3: t_k = c/sqrt(k)
    for c in [0.1,0.08,0.05,0.02,0.01]:
        # load
        loss_mom = np.load(os.path.join(datpath, "recipsqrtss", "loss_mom_" + str(c*1000) + ".npy"))
        loss_nomom = np.load(os.path.join(datpath, "recipsqrtss", "loss_nomom_" + str(c*1000) + ".npy"))
        closeness_mom = np.load(os.path.join(datpath, "recipsqrtss", "fwdbwd_mom_" + str(c*1000) + ".npy"))
        closeness_nomom = np.load(os.path.join(datpath, "recipsqrtss", "fwdbwd_nomom_" + str(c*1000) + ".npy"))
        l2_mom = np.load(os.path.join(datpath, "recipsqrtss", "l2_mom_" + str(c*1000) + ".npy"))
        l2_nomom = np.load(os.path.join(datpath, "recipsqrtss", "l2_nomom_" + str(c*1000) + ".npy"))   
        min_attain = np.amin(np.concatenate(([min_attain], loss_mom, loss_nomom)))     
        ax.plot(loss_mom, label = str(c) + " mom", marker = 'o')
        ax.plot(loss_nomom, label = str(c) + " nomom", marker = 'x')

        ax2.plot(closeness_mom, label = str(c) + " mom", marker = 'o')
        ax2.plot(closeness_nomom, label = str(c) + " nomom", marker = 'x')

        ax3.plot(loss_mom - true_min + eps, label = str(c) + " mom", marker = 'o')
        ax3.plot(loss_nomom - true_min + eps, label = str(c) + " nomom", marker = 'x')

        ax4.plot(l2_mom, label = str(c) + " mom", marker = 'o')
        ax4.plot(l2_nomom, label = str(c) + " nomom", marker = 'x')

    fig.suptitle("Loss")
    fig2.suptitle("Fwdbwd error")
    fig3.suptitle("Loss minus minimum recon")
    fig4.suptitle("L2 distance to true min")

    ax.set_ylim([true_min.item()*0.95, init_err*1.5])
    ax2.set_ylim([currminfwdbwd.item()*0.95, init_closeness_mom*1.5])
    ax3.set_ylim([(min_attain-true_min)*0.9, (init_err-true_min)*2.5])
    ax4.set_ylim([0,init_l2*1.5])

    fig.legend()
    fig2.legend() 
    fig3.legend()
    fig4.legend()
    if PDF:
        fig.savefig(os.path.join(figs_dir, "recipsqrtss_losses.pdf"))
        fig2.savefig(os.path.join(figs_dir, "recipsqrtss_fwdbwd.pdf"))
        fig3.savefig(os.path.join(figs_dir, "recipsqrtss_loglosses.pdf"))
        fig4.savefig(os.path.join(figs_dir, "recipsqrtss_l2totrue.pdf"))
    if PNG:
        fig.savefig(os.path.join(figs_dir, "recipsqrtss_losses"))
        fig2.savefig(os.path.join(figs_dir, "recipsqrtss_fwdbwd"))
        fig3.savefig(os.path.join(figs_dir, "recipsqrtss_loglosses"))
        fig4.savefig(os.path.join(figs_dir, "recipsqrtss_l2totrue"))
    end = time.time()
    print("constant ss done, elapsed time", end-start)
#%%
if MAPTRANSFER:
    start = time.time()
    fig, ax = plt.subplots(figsize = (6.4,4.8), dpi = 150) # loss plot
    fig2, ax2 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # fwdbwd plot
    fig3, ax3 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # log-loss plot 
    ax3.set_yscale('log')
    if LONGTIME:
        ax3.set_xscale('log')
        ax3.set_xlim(right = 10**3)
    ax3.plot(gdloss[:10**3]- true_min + eps, color='k', label = 'gd')
    ax3.plot(nesterovloss[:10**3]- true_min + eps, color='maroon', label = 'nesterov')
    fig4, ax4 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # l2 distance to minimum
    ax.axhline(true_min, label = "true min", linestyle = '--')
    min_attain = np.amin(np.concatenate((gdloss[:10**3], nesterovloss[:10**3])))

    # load standard
    loss_mom = np.load(os.path.join(datpath, "maptransfer", "standard", "loss_mom.npy"))
    loss_nomom = np.load(os.path.join(datpath, "maptransfer", "standard", "loss_nomom.npy"))
    closeness_mom = np.load(os.path.join(datpath, "maptransfer", "standard", "fwdbwd_mom.npy"))
    closeness_nomom = np.load(os.path.join(datpath, "maptransfer", "standard", "fwdbwd_nomom.npy"))
    l2_mom = np.load(os.path.join(datpath, "maptransfer", "standard", "l2_mom.npy"))
    l2_nomom = np.load(os.path.join(datpath, "maptransfer", "standard", "l2_nomom.npy"))
    min_attain = np.amin(np.concatenate(([min_attain], loss_mom, loss_nomom)))
    # define useful constants
    init_err = loss_mom[0]
    init_closeness_mom = closeness_mom[0]
    init_l2 = l2_mom[0]
    currminfwdbwd = np.amin(np.concatenate((closeness_mom, closeness_nomom)))



    # plot
    ax.plot(loss_mom, label = "mom", marker = 'o')
    ax.plot(loss_nomom, label = "nomom", marker = 'x')

    ax2.plot(closeness_mom, label = "mom", marker = 'o')
    ax2.plot(closeness_nomom, label = "nomom", marker = 'x')

    ax3.plot(loss_mom - true_min + eps, label = "mom", marker = 'o')
    ax3.plot(loss_nomom - true_min + eps, label = "nomom", marker = 'x')

    ax4.plot(l2_mom, label = "mom", marker = 'o')
    ax4.plot(l2_nomom, label = "nomom", marker = 'x')

    # load switched
    loss_mom = np.load(os.path.join(datpath, "maptransfer", "switch", "loss_mom.npy"))
    loss_nomom = np.load(os.path.join(datpath, "maptransfer", "switch", "loss_nomom.npy"))
    closeness_mom = np.load(os.path.join(datpath, "maptransfer", "switch", "fwdbwd_mom.npy"))
    closeness_nomom = np.load(os.path.join(datpath, "maptransfer", "switch", "fwdbwd_nomom.npy"))
    l2_mom = np.load(os.path.join(datpath, "maptransfer", "switch", "l2_mom.npy"))
    l2_nomom = np.load(os.path.join(datpath, "maptransfer", "switch", "l2_nomom.npy"))
    min_attain = np.amin(np.concatenate(([min_attain], loss_mom, loss_nomom)))
    # plot
    ax.plot(loss_mom, label = "mom switch", marker = '^')
    ax.plot(loss_nomom, label = "nomom switch", marker = 'v')

    ax2.plot(closeness_mom, label = "mom switch", marker = '^')
    ax2.plot(closeness_nomom, label = "nomom switch", marker = 'v')

    ax3.plot(loss_mom - true_min + eps, label = "mom switch", marker = '^')
    ax3.plot(loss_nomom - true_min + eps, label = "nomom switch", marker = 'v')

    ax4.plot(l2_mom, label = "mom switch", marker = '^')
    ax4.plot(l2_nomom, label = "nomom switch", marker = 'v')

    # rest of plot
    fig.suptitle("Loss")
    fig2.suptitle("Fwdbwd error")
    fig3.suptitle("Loss minus minimum recon")
    fig4.suptitle("L2 distance to true min")

    ax.set_ylim([true_min.item()*0.95, init_err*1.5])
    ax2.set_ylim([currminfwdbwd.item()*0.95, init_closeness_mom*1.5])
    ax3.set_ylim([(min_attain-true_min)*0.9, (init_err-true_min)*2.5])
    ax4.set_ylim([0,init_l2*1.5])

    fig.legend()
    fig2.legend() 
    fig3.legend()
    fig4.legend()
    if PDF:
        fig.savefig(os.path.join(figs_dir, "maptransfer_losses.pdf"))
        fig2.savefig(os.path.join(figs_dir, "maptransfer_fwdbwd.pdf"))
        fig3.savefig(os.path.join(figs_dir, "maptransfer_loglosses.pdf"))
        fig4.savefig(os.path.join(figs_dir, "maptransfer_l2totrue.pdf"))
    if PNG:
        fig.savefig(os.path.join(figs_dir, "maptransfer_losses"))
        fig2.savefig(os.path.join(figs_dir, "maptransfer_fwdbwd"))
        fig3.savefig(os.path.join(figs_dir, "maptransfer_loglosses"))
        fig4.savefig(os.path.join(figs_dir, "maptransfer_l2totrue"))

    print("map transfer done")

#%%
if DOMAINCHANGE:
    print("domain change todo")

if ALTTRANSFORM:
    MAXITERS = 1000
    # do not do l2 for this one as the true min was not updated during creation
    fig, ax = plt.subplots(figsize = (6.4,4.8), dpi = 150) # loss plot
    fig2, ax2 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # fwdbwd plot
    
    true_min = np.load(os.path.join(datpath, "true_min_val_alttransform.npy"))
    gdloss = np.load(os.path.join(datpath, "true_gd_progression_alttransform.npy"))
    #gdloss_recip = np.load(os.path.join(datpath, "recip_gd_progression.npy"))
    nesterovloss = np.load(os.path.join(datpath, "nesterov_progression_alttransform.npy"))

    fig3, ax3 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # log-loss plot 
    ax3.set_yscale('log')
    if LONGTIME:
        ax3.set_xscale('log')
        ax3.set_xlim(right = MAXITERS)
    ax3.plot(gdloss[:MAXITERS]- true_min + eps, color='k', label = 'gd')
    ax3.plot(nesterovloss[:MAXITERS]- true_min + eps, color='maroon', label = 'nesterov')
    min_attain = np.amin(np.concatenate((gdloss[:MAXITERS], nesterovloss[:MAXITERS])))
    ax.axhline(true_min, label = "true min", linestyle = '--')

    # TYPE 2: t_k = c/n
    for c in [0.1,0.08,0.05,0.02,0.01]:
        loss_mom = np.load(os.path.join(datpath, "alttransform", "loss_mom_" + str(c*1000) + ".npy"))
        loss_nomom = np.load(os.path.join(datpath, "alttransform", "loss_nomom_" + str(c*1000) + ".npy"))
        closeness_mom = np.load(os.path.join(datpath, "alttransform", "fwdbwd_mom_" + str(c*1000) + ".npy"))
        closeness_nomom = np.load(os.path.join(datpath, "alttransform", "fwdbwd_nomom_" + str(c*1000) + ".npy"))
        # l2_mom = np.load(os.path.join(datpath, "alttransform", "l2_mom_" + str(c*1000) + ".npy"))
        # l2_nomom = np.load(os.path.join(datpath, "alttransform", "l2_nomom_" + str(c*1000) + ".npy"))
        init_err = loss_mom[0]

        ax.plot(loss_mom, label = str(c) + " mom", marker = 'o')
        ax.plot(loss_nomom, label = str(c) + " nomom", marker = 'x')

        ax2.plot(closeness_mom, label = str(c) + " mom", marker = 'o')
        ax2.plot(closeness_nomom, label = str(c) + " nomom", marker = 'x')

        ax3.plot(loss_mom - true_min + eps, label = str(c) + " mom", marker = 'o')
        ax3.plot(loss_nomom - true_min + eps, label = str(c) + " nomom", marker = 'x')
    
        min_attain = np.amin(np.concatenate(([min_attain], loss_mom, loss_nomom)))
        # ax4.plot(l2_mom, label = str(c) + " mom", marker = 'o')
        # ax4.plot(l2_nomom, label = str(c) + " nomom", marker = 'x')

    fig.suptitle("Loss")
    fig2.suptitle("Fwdbwd error")
    fig3.suptitle("Loss minus minimum recon")
    # fig4.suptitle("L2 distance to true min")

    ax.set_ylim([true_min*0.95, init_err*1.1])
    ax2.set_ylim([currminfwdbwd.item()*0.95, init_closeness_mom*1.5])
    ax3.set_ylim([(min_attain-true_min)*0.9, (init_err-true_min)*2.5])
    # ax4.set_ylim([0,init_l2*1.5])

    fig.legend()
    fig2.legend() 
    fig3.legend()
    # fig4.legend()
    if PDF:
        fig.savefig(os.path.join(figs_dir, "alttrans_losses.pdf"))
        fig2.savefig(os.path.join(figs_dir, "alttrans_fwdbwd.pdf"))
        fig3.savefig(os.path.join(figs_dir, "alttrans_loglosses.pdf"))
    if PNG:
        fig.savefig(os.path.join(figs_dir, "alttrans_losses"))
        fig2.savefig(os.path.join(figs_dir, "alttrans_fwdbwd"))
        fig3.savefig(os.path.join(figs_dir, "alttrans_loglosses"))
    # fig4.savefig(os.path.join(figs_dir, "alttrans_l2totrue.pdf"))
        # save

    end = time.time()
    print("alt ray transform, elapsed time", end-start)
    

if DENOISE:
    MAXITERS = 1000
    fig, ax = plt.subplots(figsize = (6.4,4.8), dpi = 150) # loss plot
    fig2, ax2 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # fwdbwd plot
    # fig4, ax4 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # l2 distance to minimum
    # 
    true_min = np.load(os.path.join(datpath, "true_min_val_denoise.npy"))
    gdloss = np.load(os.path.join(datpath, "true_gd_progression_denoise.npy"))
    #gdloss_recip = np.load(os.path.join(datpath, "recip_gd_progression.npy"))
    nesterovloss = np.load(os.path.join(datpath, "nesterov_progression_denoise.npy"))

    fig3, ax3 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # log-loss plot 
    ax3.set_yscale('log')
    if LONGTIME:
        ax3.set_xscale('log')
        ax3.set_xlim(right = MAXITERS)
    ax3.plot(gdloss[:MAXITERS]- true_min + eps, color='k', label = 'gd')
    ax3.plot(nesterovloss[:MAXITERS]- true_min + eps, color='maroon', label = 'nesterov')
    min_attain = np.amin(np.concatenate((gdloss[:MAXITERS], nesterovloss[:MAXITERS])))
    ax.axhline(true_min, label = "true min", linestyle = '--')


    
    # TYPE 2: t_k = c/n
    for c in [0.1,0.08,0.05,0.02,0.01]:
        loss_mom = np.load(os.path.join(datpath, "denoise", "loss_mom_" + str(c*1000) + ".npy"))
        loss_nomom = np.load(os.path.join(datpath, "denoise", "loss_nomom_" + str(c*1000) + ".npy"))
        closeness_mom = np.load(os.path.join(datpath, "denoise", "fwdbwd_mom_" + str(c*1000) + ".npy"))
        closeness_nomom = np.load(os.path.join(datpath, "denoise", "fwdbwd_nomom_" + str(c*1000) + ".npy"))
        init_err = loss_mom[0]

        ax.plot(loss_mom, label = str(c) + " mom", marker = 'o')
        ax.plot(loss_nomom, label = str(c) + " nomom", marker = 'x')

        ax2.plot(closeness_mom, label = str(c) + " mom", marker = 'o')
        ax2.plot(closeness_nomom, label = str(c) + " nomom", marker = 'x')

        ax3.plot(loss_mom - true_min + eps, label = str(c) + " mom", marker = 'o')
        ax3.plot(loss_nomom - true_min + eps, label = str(c) + " nomom", marker = 'x')
    
        min_attain = np.amin(np.concatenate(([min_attain], loss_mom, loss_nomom)))

        
    fig.suptitle("Loss")
    fig2.suptitle("Fwdbwd error")
    fig3.suptitle("Loss minus minimum recon")
    # fig4.suptitle("L2 distance to true min")

    ax.set_ylim([true_min*0.95, init_err*1.1])
    ax2.set_ylim([currminfwdbwd.item()*0.95, init_closeness_mom*1.5])
    ax3.set_ylim([(min_attain-true_min)*0.9, (init_err-true_min)*2.5])
    # ax4.set_ylim([0,init_l2*1.5])

    fig.legend()
    fig2.legend() 
    fig3.legend()

    if PDF:
        fig.savefig(os.path.join(figs_dir, "denoise_losses.pdf"))
        fig2.savefig(os.path.join(figs_dir, "denoise_fwdbwd.pdf"))
        fig3.savefig(os.path.join(figs_dir, "denoise_loglosses.pdf"))
    if PNG:
        fig.savefig(os.path.join(figs_dir, "denoise_losses"))
        fig2.savefig(os.path.join(figs_dir, "denoise_fwdbwd"))
        fig3.savefig(os.path.join(figs_dir, "denoise_loglosses"))

    end = time.time()
    print("denoise, elapsed time", end-start)
# %%
if CROSS:
    start = time.time()

    true_min = np.load(os.path.join(datpath, "true_min_val.npy"))
    gdloss = np.load(os.path.join(datpath, "true_gd_progression.npy"))
    nesterovloss = np.load(os.path.join(datpath, "nesterov_progression.npy"))

    fig, ax = plt.subplots(figsize = (6.4,4.8), dpi = 150) # loss plot
    fig2, ax2 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # fwdbwd plot
    fig3, ax3 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # log-loss plot 
    ax3.set_yscale('log')
    if LONGTIME:
        ax3.set_xscale('log')
        ax3.set_xlim(right = 10**3)
    ax3.plot(gdloss[:10**3]- true_min + eps, color='k', label = 'gd')
    ax3.plot(nesterovloss[:10**3]- true_min + eps, color='maroon', label = 'nesterov')
    fig4, ax4 = plt.subplots(figsize = (6.4,4.8), dpi = 150) # l2 distance to minimum
    # bookkeeping: find minimum loss and fwdbwd
    
    ax.axhline(true_min, label = "true min", linestyle = '--')
    #ax3.axhline(true_min, label = "true min", linestyle = '--')
    print(true_min)
    init_err = gdloss[0]
    
    min_attain = np.amin(np.concatenate((gdloss[:10**3], nesterovloss[:10**3])))

    # loop 2: now plot.
    # loss_mom = np.load(os.path.join(datpath, "stepsizeext", "loss_mom_" + "recip" + ".npy"))
    # loss_nomom = np.load(os.path.join(datpath, "stepsizeext", "loss_nomom_" + "recip" + ".npy"))
    # closeness_mom = np.load(os.path.join(datpath, "stepsizeext", "fwdbwd_mom_" + "recip" + ".npy"))
    # closeness_nomom = np.load(os.path.join(datpath, "stepsizeext", "fwdbwd_nomom_" + "recip" + ".npy"))
    # l2_mom = np.load(os.path.join(datpath, "stepsizeext", "l2_mom_" + "recip" + ".npy"))
    # l2_nomom = np.load(os.path.join(datpath, "stepsizeext", "l2_nomom_" + "recip" + ".npy"))

    # min_attain = np.amin(np.concatenate(([min_attain], loss_mom, loss_nomom)))

    # ax.plot(loss_mom, label = "deconv mom", marker = 'o')
    # ax.plot(loss_nomom, label = "deconv nomom", marker = 'x')

    # ax2.plot(closeness_mom, label = "deconv mom", marker = 'o')
    # ax2.plot(closeness_nomom, label = "deconv nomom", marker = 'x')

    # ax3.plot(loss_mom - true_min + eps, label = "deconv mom", marker = 'o')
    # ax3.plot(loss_nomom - true_min + eps, label = "deconv nomom", marker = 'x')

    # ax4.plot(l2_mom, label = "deconv mom", marker = 'o')
    # ax4.plot(l2_nomom, label = "deconv nomom", marker = 'x')

    if MODE == "ELLIPSE":
        figs_dir_temp = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/figs/ellipse"
        datpath_temp = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/datELLIPSE"
    else:
        figs_dir_temp = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/figs/lodopab"
        datpath_temp = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/datLODOPAB"

    if LONGTIME:
        figs_dir_temp = figs_dir_temp + "_longtime"
        datpath_temp = datpath_temp+"long"



    # TYPE 2: t_k = c/k
    for c in [0.1,0.08,0.05,0.02,0.01]:
        # load
        loss_mom = np.load(os.path.join(datpath, "recipss", "loss_mom_" + str(c*1000) + ".npy"))
        loss_nomom = np.load(os.path.join(datpath, "recipss", "loss_nomom_" + str(c*1000) + ".npy"))
        closeness_mom = np.load(os.path.join(datpath, "recipss", "fwdbwd_mom_" + str(c*1000) + ".npy"))
        closeness_nomom = np.load(os.path.join(datpath, "recipss", "fwdbwd_nomom_" + str(c*1000) + ".npy"))
        l2_mom = np.load(os.path.join(datpath, "recipss", "l2_mom_" + str(c*1000) + ".npy"))
        l2_nomom = np.load(os.path.join(datpath, "recipss", "l2_nomom_" + str(c*1000) + ".npy"))
        min_attain = np.amin(np.concatenate(([min_attain], loss_mom, loss_nomom)))        
        ax.plot(loss_mom, label = str(c) + " deconvmom", color='blue')
        ax.plot(loss_nomom, label = str(c) + " deconvnomom", color='green')

        ax2.plot(closeness_mom, label = str(c) + " deconvmom", color='blue')
        ax2.plot(closeness_nomom, label = str(c) + " deconvnomom", color='green')

        ax3.plot(loss_mom - true_min + eps, label = str(c) + " deconvmom", color='blue')
        ax3.plot(loss_nomom - true_min + eps, label = str(c) + " deconvnomom", color='green')

        ax4.plot(l2_mom, label = str(c) + " deconvmom", color='blue')
        ax4.plot(l2_nomom, label = str(c) + " deconvnomom", color='green')
        min_attain = np.amin(np.concatenate(([min_attain], loss_mom, loss_nomom)))

    for c in [0.1,0.08,0.05,0.02,0.01]:
        loss_mom = np.load(os.path.join(datpath_temp, "deconv", "loss_mom_" + str(c*1000) + ".npy"))
        loss_nomom = np.load(os.path.join(datpath_temp, "deconv", "loss_nomom_" + str(c*1000) + ".npy"))
        closeness_mom = np.load(os.path.join(datpath_temp, "deconv", "fwdbwd_mom_" + str(c*1000) + ".npy"))
        closeness_nomom = np.load(os.path.join(datpath_temp, "deconv", "fwdbwd_nomom_" + str(c*1000) + ".npy"))
        l2_mom = np.load(os.path.join(datpath_temp, "deconv", "l2_mom_" + str(c*1000) + ".npy"))
        l2_nomom = np.load(os.path.join(datpath_temp, "deconv", "l2_nomom_" + str(c*1000) + ".npy"))
        init_l2 = l2_mom[0]
        ax.plot(loss_mom, label = str(c) + " fbpmom", color='red')
        ax.plot(loss_nomom, label = str(c) + " fbpnomom", color='yellow')

        ax2.plot(closeness_mom, label = str(c) + " fbpmom", color='red')
        ax2.plot(closeness_nomom, label = str(c) + " fbpnomom", color='yellow')

        ax3.plot(loss_mom - true_min + eps, label = str(c) + " fbpmom", color='red')
        ax3.plot(loss_nomom - true_min + eps, label = str(c) + " fbpnomom", color='yellow')
    
        ax4.plot(l2_mom, label = str(c) + " fbpmom", color='red')
        ax4.plot(l2_nomom, label = str(c) + " fbpnomom", color='yellow')

        min_attain = np.amin(np.concatenate(([min_attain], loss_mom, loss_nomom)))
    fig.suptitle("Loss")
    fig2.suptitle("Fwdbwd error")
    fig3.suptitle("Loss minus minimum recon")
    fig4.suptitle("L2 distance to true min")

    ax.set_ylim([true_min.item()*0.95, init_err*1.5])
    ax2.set_ylim([currminfwdbwd.item()*0.95, init_closeness_mom*1.5])
    ax3.set_ylim([(min_attain-true_min)*0.9, (init_err-true_min)*2.5])
    ax4.set_ylim([0,init_l2*1.5])

    fig.legend()
    fig2.legend() 
    fig3.legend(fontsize = 'x-small', loc=3)
    fig4.legend()
    if PDF:
        fig.savefig(os.path.join(figs_dir, "comb_deconv_losses.pdf"))
        fig2.savefig(os.path.join(figs_dir, "comb_deconv_fwdbwd.pdf"))
        fig3.savefig(os.path.join(figs_dir, "comb_deconv_loglosses.pdf"))
        fig4.savefig(os.path.join(figs_dir, "comb_deconv_l2totrue.pdf"))
    if PNG:
        fig.savefig(os.path.join(figs_dir, "comb_deconv_losses"))
        fig2.savefig(os.path.join(figs_dir, "comb_deconv_fwdbwd"))
        fig3.savefig(os.path.join(figs_dir, "comb_deconv_loglosses"))
        fig4.savefig(os.path.join(figs_dir, "comb_deconv_l2totrue"))
    end = time.time()
    print("adaptive stepsize extension done, elapsed time", end-start)