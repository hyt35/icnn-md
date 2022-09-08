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
eps=1e-6
#%% EXPERIMENT FLAGS
if EXPTYPE == "ALL":
    STEPSIZEEXT = True
    CONSTSS = True
    MAPTRANSFER = True
    DOMAINCHANGE = True
    
else: # manual
    STEPSIZEEXT = True
    CONSTSS = True
    MAPTRANSFER = False
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
    datpath = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/datELLIPSE"
else:
    figs_dir = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/figs/lodopab"
    datpath = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/datLODOPAB"

if LONGTIME:
    figs_dir = figs_dir + "_longtime"
    datpath = datpath+"long"
args=parse_import.parse_commandline_args()
true_min = np.load(os.path.join(datpath, "true_min_val.npy"))
gdloss = np.load(os.path.join(datpath, "true_gd_progression.npy"))
#gdloss_recip = np.load(os.path.join(datpath, "recip_gd_progression.npy"))
nesterovloss = np.load(os.path.join(datpath, "nesterov_progression.npy"))
if STEPSIZEEXT:
    # Experiment: step size extension with different extension methods.
    start = time.time()

    fig, ax = plt.subplots(figsize = (8,6), dpi = 150) # loss plot
    fig2, ax2 = plt.subplots(figsize = (8,6), dpi = 150) # fwdbwd plot
    fig3, ax3 = plt.subplots(figsize = (8,6), dpi = 150) # log-loss plot 
    ax3.set_yscale('log')
    if LONGTIME:
        ax3.set_xscale('log')
        ax3.set_xlim(right = 10**3)
    ax3.plot(gdloss[:10**3]- true_min + eps, color='k', label = 'gd')
    ax3.plot(nesterovloss[:10**3]- true_min + eps, color='maroon', label = 'nesterov')
    fig4, ax4 = plt.subplots(figsize = (8,6), dpi = 150) # l2 distance to minimum
    # bookkeeping: find minimum loss and fwdbwd
    # currminloss = None
    currminfwdbwd = None
    init_err = None
    init_closeness_mom = None
    init_l2 = None
    
    ax.axhline(true_min, label = "true min", linestyle = '--')
    #ax3.axhline(true_min, label = "true min", linestyle = '--')
    print(true_min)
    # for extend_type in ["max", "mean", "min", "final", "recip"]:
    #     loss_mom = np.load(os.path.join(datpath, "stepsizeext", "loss_mom_" + extend_type + ".npy"))
    #     loss_nomom = np.load(os.path.join(datpath, "stepsizeext", "loss_nomom_" + extend_type + ".npy"))
    #     closeness_mom = np.load(os.path.join(datpath, "stepsizeext", "fwdbwd_mom_" + extend_type + ".npy"))
    #     closeness_nomom = np.load(os.path.join(datpath, "stepsizeext", "fwdbwd_nomom_" + extend_type + ".npy"))
    #     l2_mom = np.load(os.path.join(datpath, "stepsizeext", "l2_mom_" + extend_type + ".npy"))
    #     l2_nomom = np.load(os.path.join(datpath, "stepsizeext", "l2_nomom_" + extend_type + ".npy"))
    #     # if currminloss is None:
    #     #     currminloss = np.amin(np.concatenate((loss_mom, loss_nomom)))
    #     # else:
    #     #     currminloss = np.amin((currminloss, np.amin(np.concatenate((loss_mom, loss_nomom)))))
    #     if currminfwdbwd is None:
    #         currminfwdbwd = np.amin(np.concatenate((closeness_mom, closeness_nomom)))
    #     else:
    #         currminfwdbwd = np.amin((currminfwdbwd, np.amin(np.concatenate((closeness_mom, closeness_nomom)))))
    #     if init_err is None:
    #         init_err = loss_mom[0]
    #     if init_closeness_mom is None:
    #         init_closeness_mom = closeness_mom[0]
    #     if init_l2 is None:
    #         init_l2 = l2_mom[0]
    #     break

    # loop 2: now plot.
    for extend_type in ["max", "mean", "min", "final", "recip"]:
        loss_mom = np.load(os.path.join(datpath, "stepsizeext", "loss_mom_" + extend_type + ".npy"))
        loss_nomom = np.load(os.path.join(datpath, "stepsizeext", "loss_nomom_" + extend_type + ".npy"))
        closeness_mom = np.load(os.path.join(datpath, "stepsizeext", "fwdbwd_mom_" + extend_type + ".npy"))
        closeness_nomom = np.load(os.path.join(datpath, "stepsizeext", "fwdbwd_nomom_" + extend_type + ".npy"))
        l2_mom = np.load(os.path.join(datpath, "stepsizeext", "l2_mom_" + extend_type + ".npy"))
        l2_nomom = np.load(os.path.join(datpath, "stepsizeext", "l2_nomom_" + extend_type + ".npy"))


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
    ax3.set_ylim(top = (init_err-true_min)*2.5)
    ax4.set_ylim([0,init_l2*1.5])

    fig.legend()
    fig2.legend() 
    fig3.legend()
    fig4.legend()
    fig.savefig(os.path.join(figs_dir, "stepsize_ext_losses"))
    fig2.savefig(os.path.join(figs_dir, "stepsize_ext_fwdbwd"))
    fig3.savefig(os.path.join(figs_dir, "stepsize_ext_loglosses"))
    fig4.savefig(os.path.join(figs_dir, "stepsize_ext_l2totrue"))
    end = time.time()
    print("adaptive stepsize extension done, elapsed time", end-start)

#%%
if CONSTSS:
    start = time.time()
    fig, ax = plt.subplots(figsize = (8,6), dpi = 150) # loss plot
    fig2, ax2 = plt.subplots(figsize = (8,6), dpi = 150) # fwdbwd plot
    fig3, ax3 = plt.subplots(figsize = (8,6), dpi = 150) # log-loss plot 
    ax3.set_yscale('log')
    if LONGTIME:
        ax3.set_xscale('log')
        ax3.set_xlim(right = 10**3)
    ax3.plot(gdloss[:10**3]- true_min + eps, color='k', label = 'gd')
    ax3.plot(nesterovloss[:10**3]- true_min + eps, color='maroon', label = 'nesterov')
    fig4, ax4 = plt.subplots(figsize = (8,6), dpi = 150) # l2 distance to minimum

    ax.axhline(true_min, label = "true min", linestyle = '--')


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
    ax3.set_ylim(top = (init_err-true_min)*2.5)
    ax4.set_ylim([0,init_l2*1.5])

    fig.legend()
    fig2.legend() 
    fig3.legend()
    fig4.legend()
    fig.savefig(os.path.join(figs_dir, "constss_losses"))
    fig2.savefig(os.path.join(figs_dir, "constss_fwdbwd"))
    fig3.savefig(os.path.join(figs_dir, "constss_loglosses"))
    fig4.savefig(os.path.join(figs_dir, "constss_l2totrue"))

    fig, ax = plt.subplots(figsize = (8,6), dpi = 150) # loss plot
    fig2, ax2 = plt.subplots(figsize = (8,6), dpi = 150) # fwdbwd plot
    fig3, ax3 = plt.subplots(figsize = (8,6), dpi = 150) # log-loss plot 
    ax3.set_yscale('log')
    if LONGTIME:
        ax3.set_xscale('log')
        ax3.set_xlim(right = 10**3)
    ax3.plot(gdloss[:10**3]- true_min + eps, color='k', label = 'gd')
    ax3.plot(nesterovloss[:10**3]- true_min + eps, color='maroon', label = 'nesterov')
    fig4, ax4 = plt.subplots(figsize = (8,6), dpi = 150) # l2 distance to minimum

    ax.axhline(true_min, label = "true min", linestyle = '--')


    # TYPE 2: t_k = c/n
    for c in [0.1,0.08,0.05,0.02,0.01]:
        # load
        loss_mom = np.load(os.path.join(datpath, "recipss", "loss_mom_" + str(c*1000) + ".npy"))
        loss_nomom = np.load(os.path.join(datpath, "recipss", "loss_nomom_" + str(c*1000) + ".npy"))
        closeness_mom = np.load(os.path.join(datpath, "recipss", "fwdbwd_mom_" + str(c*1000) + ".npy"))
        closeness_nomom = np.load(os.path.join(datpath, "recipss", "fwdbwd_nomom_" + str(c*1000) + ".npy"))
        l2_mom = np.load(os.path.join(datpath, "recipss", "l2_mom_" + str(c*1000) + ".npy"))
        l2_nomom = np.load(os.path.join(datpath, "recipss", "l2_nomom_" + str(c*1000) + ".npy"))        
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
    ax3.set_ylim(top = (init_err-true_min)*2.5)
    ax4.set_ylim([0,init_l2*1.5])

    fig.legend()
    fig2.legend() 
    fig3.legend()
    fig4.legend()
    fig.savefig(os.path.join(figs_dir, "recipss_losses"))
    fig2.savefig(os.path.join(figs_dir, "recipss_fwdbwd"))
    fig3.savefig(os.path.join(figs_dir, "recipss_loglosses"))
    fig4.savefig(os.path.join(figs_dir, "recipss_l2totrue"))
    end = time.time()
    print("constant ss done, elapsed time", end-start)
#%%
if MAPTRANSFER:
    start = time.time()
    fig, ax = plt.subplots(figsize = (8,6), dpi = 150) # loss plot
    fig2, ax2 = plt.subplots(figsize = (8,6), dpi = 150) # fwdbwd plot
    fig3, ax3 = plt.subplots(figsize = (8,6), dpi = 150) # log-loss plot 
    ax3.set_yscale('log')
    if LONGTIME:
        ax3.set_xscale('log')
        ax3.set_xlim(right = 10**3)
    ax3.plot(gdloss[:10**3]- true_min + eps, color='k', label = 'gd')
    ax3.plot(nesterovloss[:10**3]- true_min + eps, color='maroon', label = 'nesterov')
    fig4, ax4 = plt.subplots(figsize = (8,6), dpi = 150) # l2 distance to minimum
    ax.axhline(true_min, label = "true min", linestyle = '--')


    # load standard
    loss_mom = np.load(os.path.join(datpath, "maptransfer", "standard", "loss_mom.npy"))
    loss_nomom = np.load(os.path.join(datpath, "maptransfer", "standard", "loss_nomom.npy"))
    closeness_mom = np.load(os.path.join(datpath, "maptransfer", "standard", "fwdbwd_mom.npy"))
    closeness_nomom = np.load(os.path.join(datpath, "maptransfer", "standard", "fwdbwd_nomom.npy"))
    l2_mom = np.load(os.path.join(datpath, "maptransfer", "standard", "l2_mom.npy"))
    l2_nomom = np.load(os.path.join(datpath, "maptransfer", "standard", "l2_nomom.npy"))

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
    ax3.set_ylim(top = (init_err-true_min)*2.5)
    ax4.set_ylim([0,init_l2*1.5])

    fig.legend()
    fig2.legend() 
    fig3.legend()
    fig4.legend()
    fig.savefig(os.path.join(figs_dir, "maptransfer_losses"))
    fig2.savefig(os.path.join(figs_dir, "maptransfer_fwdbwd"))
    fig3.savefig(os.path.join(figs_dir, "maptransfer_loglosses"))
    fig4.savefig(os.path.join(figs_dir, "maptransfer_l2totrue"))

    print("map transfer done")

#%%
if DOMAINCHANGE:
    print("domain change todo")

