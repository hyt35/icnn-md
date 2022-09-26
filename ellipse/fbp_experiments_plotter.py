# -*- coding: utf-8 -*-
"""
Created on Fri Sep  13:44:37 2022

@author: hongy
"""
#from icnn import DenseICGN
#from denoising_nets_for_mcmc import ICNN
# from pickle import TRUE
# from sqlite3 import TimeFromTicks
# from xxlimited import foo
import numpy as np
# from odl.tomo.analytic.filtered_back_projection import fbp_op
# import torch
# from torch._C import LoggerBase
# import torch.nn as nn
# import torch.autograd as autograd
# import torchvision
import matplotlib as mpl
import matplotlib.pyplot as plt
#from iunets import iUNet
import parse_import
# import logging
# from datetime import datetime
# import torch.nn.functional as F
# from models import ICNNCoupleMomentum, ICNNCouple
# from pathlib import Path
# from dival import get_standard_dataset
# from dival.datasets.fbp_dataset import get_cached_fbp_dataset
import tensorflow as tf
import os
# import odl
# from tqdm import tqdm
# import torch_wrapper
# from torch.utils import tensorboard
# from torchvision.utils import make_grid, save_image
import time
# from odl.ufunc_ops.ufunc_ops import log_op
# from dival.evaluation import TaskTable
# from dival.measure import PSNR, SSIM
# from dival.reconstructors.odl_reconstructors import FBPReconstructor
# from dival.datasets.standard import get_standard_dataset
# from dival.util.constants import MU_MAX
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
#%% EXPERIMENT PARAMETERS
IMPL = 'astra_cuda'
MODE = "ELLIPSE"
# MODE = "LODOPAB"
EXPTYPE = "MAN" # experiment type 
LONGTIME = True
MOMSPLIT = False
PDF = True
PNG = True
TITLE = False
eps=1e-6
#%% EXPERIMENT FLAGS
if EXPTYPE == "ALL":
    STEPSIZEEXT = True
    CONSTSS = True
    MAPTRANSFER = True
    DOMAINCHANGE = True
    ALTTRANSFORM = True
    DECONV = True
    
else: # manual
    STEPSIZEEXT = True
    CONSTSS = True
    MAPTRANSFER = True
    DOMAINCHANGE = False
    ALTTRANSFORM = True
    DECONV = False
#%% MATPLOTLIB FLAGS
mpl.rcParams['xtick.labelsize'] = "large"
mpl.rcParams['ytick.labelsize'] = "large"
mpl.rcParams['axes.labelsize'] = "large"
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
#%%
tf.io.gfile.makedirs(figs_dir)
#%%
args=parse_import.parse_commandline_args()
true_min = np.load(os.path.join(datpath, "true_min_val.npy"))
gdloss = np.load(os.path.join(datpath, "true_gd_progression.npy"))
#gdloss_recip = np.load(os.path.join(datpath, "recip_gd_progression.npy"))
nesterovloss = np.load(os.path.join(datpath, "nesterov_progression.npy"))

_gdline = mlines.Line2D([], [], color='black', linestyle='solid', label='GD')
_nestline = mlines.Line2D([], [], color='green', linestyle='solid', label='Nesterov')
_lmdline = mlines.Line2D([], [], color='blue',  linestyle='dotted', label='LMD')
_lamdline = mlines.Line2D([], [], color='red',  linestyle='dashed', label='LAMD')


if STEPSIZEEXT:
    # Experiment: step size extension with different extension methods.
    start = time.time()

    fig, ax = plt.subplots(figsize = (8.0,4.8), dpi = 150) # loss plot
    fig2, ax2 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # fwdbwd plot
    fig3, ax3 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # log-loss plot 
    ax3.set_yscale('log')
    if LONGTIME:
        ax3.set_xscale('log')
        ax3.set_xlim(right = 10**3)
    ax3.plot(gdloss[:10**3]- true_min + eps, color='k', label = 'gd')
    ax3.plot(nesterovloss[:10**3]- true_min + eps, color='green', label = 'nesterov')
    fig4, ax4 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # l2 distance to minimum
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
    for extend_type, marker in zip(["max", "mean", "min", "final", "recip"], ['o','x','+', '^', '*']):
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

        ax.plot(loss_mom, label = str(extend_type) + " mom", marker = marker, markevery=100, linestyle = "dashed", color='red')
        ax.plot(loss_nomom[~np.isinf(loss_nomom)], label = str(extend_type) + " nomom", marker = marker, markevery=100, linestyle = "dotted", color='blue')

        ax2.plot(closeness_mom, label = str(extend_type) + " mom", marker = marker, markevery=100, linestyle = "dashed", color='red')
        ax2.plot(closeness_nomom, label = str(extend_type) + " nomom", marker = marker, markevery=100, linestyle = "dotted", color='blue')

        ax3.loglog(loss_mom - true_min + eps, label = str(extend_type) + " mom", marker = marker, markevery=0.1, linestyle = "dashed", color='red')
        ax3.loglog(loss_nomom[~np.isinf(loss_nomom)] - true_min + eps, label = str(extend_type) + " nomom", marker = marker, markevery=0.1, linestyle = "dotted", color='blue')

        ax4.plot(l2_mom, label = str(extend_type) + " mom", marker = marker, markevery=100, linestyle = "dashed", color='red')
        ax4.plot(l2_nomom, label = str(extend_type) + " nomom", marker = marker, markevery=100, linestyle = "dotted", color='blue')

    if TITLE:   
        fig.suptitle("Loss")
        fig2.suptitle("Fwdbwd error")
        fig3.suptitle("Loss minus minimum recon")
        fig4.suptitle("L2 distance to true min")

    ax.set_ylim([true_min.item()*0.95, init_err*1.5])
    ax2.set_ylim([currminfwdbwd.item()*0.95, init_closeness_mom*1.5])
    ax3.set_ylim([(min_attain-true_min)*0.9, (init_err-true_min)*2.5])
    ax4.set_ylim([0,init_l2*1.5])

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$f(x_k)$")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"$\|(\nabla M_\theta^* \circ \nabla M_\theta - I) (x_k)\|$")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel(r"$\|f(x_k) - f^*\|$")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel(r"$\|x_k - x^*\|$")

    # fig.legend()
    # fig2.legend() 
    # fig3.legend()
    # fig4.legend()

    _gdline = mlines.Line2D([], [], color='black', linestyle='solid', label='GD')
    _nestline = mlines.Line2D([], [], color='green', linestyle='solid', label='Nesterov')
    _lmdline = mlines.Line2D([], [], color='blue',  linestyle='dotted', label='LMD')
    _lamdline = mlines.Line2D([], [], color='red',  linestyle='dashed', label='LAMD')

    #_mrkempty =  mlines.Line2D([], [], color='black', marker='', linestyle='None',label='Method')
    _mrk1 = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                            markersize=10, label='Max')
    _mrk2 = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                            markersize=10, label='Mean')
    _mrk3 = mlines.Line2D([], [], color='black', marker='+', linestyle='None',
                            markersize=10, label='Min')
    _mrk4 = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                            markersize=10, label='Final')
    _mrk5 = mlines.Line2D([], [], color='black', marker='*', linestyle='None',
                            markersize=10, label='Recip')

    for axfoo in [ax, ax2, ax3, ax4]:
        box = axfoo.get_position()
        axfoo.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        legend1 = axfoo.legend(handles=[_gdline, _nestline, _lmdline, _lamdline], bbox_to_anchor=(1.02, 0.7),
                                loc='center left', borderaxespad=0.)
        axfoo.legend(handles=[_mrk1,_mrk2,_mrk3,_mrk4,_mrk5], bbox_to_anchor=(1.02,0.3),
                                loc='center left', borderaxespad=0.)
        axfoo.add_artist(legend1)

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
    fig, ax = plt.subplots(figsize = (8.0,4.8), dpi = 150) # loss plot
    fig2, ax2 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # fwdbwd plot
    fig3, ax3 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # log-loss plot 
    ax3.set_yscale('log')
    if LONGTIME:
        ax3.set_xscale('log')
        ax3.set_xlim(right = 10**3)
    ax3.plot(gdloss[:10**3]- true_min + eps, color='k', label = 'gd')
    ax3.plot(nesterovloss[:10**3]- true_min + eps, color='green', label = 'nesterov')
    fig4, ax4 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # l2 distance to minimum

    ax.axhline(true_min, label = "true min", linestyle = '--')
    min_attain = np.amin(np.concatenate((gdloss[:10**3], nesterovloss[:10**3])))

    init_err = None
    init_closeness_mom = None
    init_l2 = None
    # TYPE 1: t_k = c
    for c, marker in zip([0.1,0.05,0.01,0.005,0.001], ['o','x','+', '^', '*']):
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


        ax.plot(loss_mom, marker = marker, markevery=100, linestyle = "dashed", color='red')
        ax.plot(loss_nomom[~np.isinf(loss_nomom)], marker = marker, markevery=100, linestyle = "dotted", color='blue')

        ax2.plot(closeness_mom, marker = marker, markevery=100, linestyle = "dashed", color='red')
        ax2.plot(closeness_nomom, marker = marker, markevery=100, linestyle = "dotted", color='blue')

        ax3.loglog(loss_mom - true_min + eps, marker = marker, markevery=0.1, linestyle = "dashed", color='red')
        ax3.loglog(loss_nomom[~np.isinf(loss_nomom)] - true_min + eps, marker = marker, markevery=0.1, linestyle = "dotted", color='blue')

        ax4.plot(l2_mom, marker = marker, markevery=100, linestyle = "dashed", color='red')
        ax4.plot(l2_nomom, marker = marker, markevery=100, linestyle = "dotted", color='blue')

    if TITLE:
        fig.suptitle("Loss")
        fig2.suptitle("Fwdbwd error")
        fig3.suptitle("Loss minus minimum recon")
        fig4.suptitle("L2 distance to true min")

    ax.set_ylim([true_min.item()*0.95, init_err*1.5])
    ax2.set_ylim([currminfwdbwd.item()*0.95, init_closeness_mom*1.5])
    ax3.set_ylim([(min_attain-true_min)*0.9, (init_err-true_min)*2.5])
    ax4.set_ylim([0,init_l2*1.5])

    
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$f(x_k)$")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"$\|(\nabla M_\theta^* \circ \nabla M_\theta - I) (x_k)\|$")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel(r"$\|f(x_k) - f^*\|$")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel(r"$\|x_k - x^*\|$")

    _mrkempty =  mlines.Line2D([], [], color='black', marker='', linestyle='None',label=r'$c$')
    _mrk1 = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                            markersize=10, label='0.1')
    _mrk2 = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                            markersize=10, label='0.05')
    _mrk3 = mlines.Line2D([], [], color='black', marker='+', linestyle='None',
                            markersize=10, label='0.01')
    _mrk4 = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                            markersize=10, label='0.005')
    _mrk5 = mlines.Line2D([], [], color='black', marker='*', linestyle='None',
                            markersize=10, label='0.001')

    for axfoo in [ax, ax2, ax3, ax4]:
        box = axfoo.get_position()
        axfoo.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        legend1 = axfoo.legend(handles=[_gdline, _nestline, _lmdline, _lamdline], bbox_to_anchor=(1.02, 0.7),
                                loc='center left', borderaxespad=0.)
        axfoo.legend(handles=[_mrkempty, _mrk1,_mrk2,_mrk3,_mrk4,_mrk5], bbox_to_anchor=(1.02,0.3),
                                loc='center left', borderaxespad=0.)
        axfoo.add_artist(legend1)


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
    fig, ax = plt.subplots(figsize = (8.0,4.8), dpi = 150) # loss plot
    fig2, ax2 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # fwdbwd plot
    fig3, ax3 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # log-loss plot 
    ax3.set_yscale('log')
    if LONGTIME:
        ax3.set_xscale('log')
        ax3.set_xlim(right = 10**3)
    ax3.plot(gdloss[:10**3]- true_min + eps, color='k', label = 'gd')
    ax3.plot(nesterovloss[:10**3]- true_min + eps, color='green', label = 'nesterov')
    fig4, ax4 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # l2 distance to minimum

    ax.axhline(true_min, label = "true min", linestyle = '--')
    min_attain = np.amin(np.concatenate((gdloss[:10**3], nesterovloss[:10**3])))

    # TYPE 2: t_k = c/k
    for c, marker in zip([0.1,0.08,0.05,0.02,0.01],['o','x','+', '^', '*']):
        # load
        loss_mom = np.load(os.path.join(datpath, "recipss", "loss_mom_" + str(c*1000) + ".npy"))
        loss_nomom = np.load(os.path.join(datpath, "recipss", "loss_nomom_" + str(c*1000) + ".npy"))
        closeness_mom = np.load(os.path.join(datpath, "recipss", "fwdbwd_mom_" + str(c*1000) + ".npy"))
        closeness_nomom = np.load(os.path.join(datpath, "recipss", "fwdbwd_nomom_" + str(c*1000) + ".npy"))
        l2_mom = np.load(os.path.join(datpath, "recipss", "l2_mom_" + str(c*1000) + ".npy"))
        l2_nomom = np.load(os.path.join(datpath, "recipss", "l2_nomom_" + str(c*1000) + ".npy"))
        min_attain = np.amin(np.concatenate(([min_attain], loss_mom, loss_nomom)))        
        ax.plot(loss_mom, marker = marker, markevery=100, linestyle = "dashed", color='red')
        ax.plot(loss_nomom[~np.isinf(loss_nomom)], marker = marker, markevery=100, linestyle = "dotted", color='blue')

        ax2.plot(closeness_mom, marker = marker, markevery=100, linestyle = "dashed", color='red')
        ax2.plot(closeness_nomom, marker = marker, markevery=100, linestyle = "dotted", color='blue')

        ax3.loglog(loss_mom - true_min + eps, marker = marker, markevery=0.1, linestyle = "dashed", color='red')
        ax3.loglog(loss_nomom[~np.isinf(loss_nomom)] - true_min + eps, marker = marker, markevery=0.1, linestyle = "dotted", color='blue')

        ax4.plot(l2_mom, marker = marker, markevery=100, linestyle = "dashed", color='red')
        ax4.plot(l2_nomom, marker = marker, markevery=100, linestyle = "dotted", color='blue')

    if TITLE:
        fig.suptitle("Loss")
        fig2.suptitle("Fwdbwd error")
        fig3.suptitle("Loss minus minimum recon")
        fig4.suptitle("L2 distance to true min")

    ax.set_ylim([true_min.item()*0.95, init_err*1.5])
    ax2.set_ylim([currminfwdbwd.item()*0.95, init_closeness_mom*1.5])
    ax3.set_ylim([(min_attain-true_min)*0.9, (init_err-true_min)*2.5])
    ax4.set_ylim([0,init_l2*1.5])

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$f(x_k)$")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"$\|(\nabla M_\theta^* \circ \nabla M_\theta - I) (x_k)\|$")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel(r"$\|f(x_k) - f^*\|$")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel(r"$\|x_k - x^*\|$")

    _mrkempty =  mlines.Line2D([], [], color='black', marker='', linestyle='None',label=r'$c$')
    _mrk1 = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                            markersize=10, label='0.1')
    _mrk2 = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                            markersize=10, label='0.08')
    _mrk3 = mlines.Line2D([], [], color='black', marker='+', linestyle='None',
                            markersize=10, label='0.05')
    _mrk4 = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                            markersize=10, label='0.02')
    _mrk5 = mlines.Line2D([], [], color='black', marker='*', linestyle='None',
                            markersize=10, label='0.01')

    for axfoo in [ax, ax2, ax3, ax4]:
        box = axfoo.get_position()
        axfoo.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        legend1 = axfoo.legend(handles=[_gdline, _nestline, _lmdline, _lamdline], bbox_to_anchor=(1.02, 0.7),
                                loc='center left', borderaxespad=0.)
        axfoo.legend(handles=[_mrkempty, _mrk1,_mrk2,_mrk3,_mrk4,_mrk5], bbox_to_anchor=(1.02,0.3),
                                loc='center left', borderaxespad=0.)
        axfoo.add_artist(legend1)
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
    fig, ax = plt.subplots(figsize = (8.0,4.8), dpi = 150) # loss plot
    fig2, ax2 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # fwdbwd plot
    fig3, ax3 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # log-loss plot 
    ax3.set_yscale('log')
    if LONGTIME:
        ax3.set_xscale('log')
        ax3.set_xlim(right = 10**3)
    ax3.plot(gdloss[:10**3]- true_min + eps, color='k', label = 'gd')
    ax3.plot(nesterovloss[:10**3]- true_min + eps, color='green', label = 'nesterov')
    fig4, ax4 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # l2 distance to minimum

    ax.axhline(true_min, label = "true min", linestyle = '--')
    min_attain = np.amin(np.concatenate((gdloss[:10**3], nesterovloss[:10**3])))

    # TYPE 3: t_k = c/sqrt(k)
    for c, marker in zip([0.1,0.08,0.05,0.02,0.01], ['o','x','+', '^', '*']):
        # load
        loss_mom = np.load(os.path.join(datpath, "recipsqrtss", "loss_mom_" + str(c*1000) + ".npy"))
        loss_nomom = np.load(os.path.join(datpath, "recipsqrtss", "loss_nomom_" + str(c*1000) + ".npy"))
        closeness_mom = np.load(os.path.join(datpath, "recipsqrtss", "fwdbwd_mom_" + str(c*1000) + ".npy"))
        closeness_nomom = np.load(os.path.join(datpath, "recipsqrtss", "fwdbwd_nomom_" + str(c*1000) + ".npy"))
        l2_mom = np.load(os.path.join(datpath, "recipsqrtss", "l2_mom_" + str(c*1000) + ".npy"))
        l2_nomom = np.load(os.path.join(datpath, "recipsqrtss", "l2_nomom_" + str(c*1000) + ".npy"))   
        min_attain = np.amin(np.concatenate(([min_attain], loss_mom, loss_nomom)))     
        ax.plot(loss_mom, marker = marker, markevery=100, linestyle = "dashed", color='red')
        ax.plot(loss_nomom[~np.isinf(loss_nomom)], marker = marker, markevery=100, linestyle = "dotted", color='blue')

        ax2.plot(closeness_mom, marker = marker, markevery=100, linestyle = "dashed", color='red')
        ax2.plot(closeness_nomom, marker = marker, markevery=100, linestyle = "dotted", color='blue')

        ax3.loglog(loss_mom - true_min + eps, marker = marker, markevery=0.1, linestyle = "dashed", color='red')
        ax3.loglog(loss_nomom[~np.isinf(loss_nomom)] - true_min + eps, marker = marker, markevery=0.1, linestyle = "dotted", color='blue')

        ax4.plot(l2_mom, marker = marker, markevery=100, linestyle = "dashed", color='red')
        ax4.plot(l2_nomom, marker = marker, markevery=100, linestyle = "dotted", color='blue')

    if TITLE:
        fig.suptitle("Loss")
        fig2.suptitle("Fwdbwd error")
        fig3.suptitle("Loss minus minimum recon")
        fig4.suptitle("L2 distance to true min")

    ax.set_ylim([true_min.item()*0.95, init_err*1.5])
    ax2.set_ylim([currminfwdbwd.item()*0.95, init_closeness_mom*1.5])
    ax3.set_ylim([(min_attain-true_min)*0.9, (init_err-true_min)*2.5])
    ax4.set_ylim([0,init_l2*1.5])

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$f(x_k)$")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"$\|(\nabla M_\theta^* \circ \nabla M_\theta - I) (x_k)\|$")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel(r"$\|f(x_k) - f^*\|$")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel(r"$\|x_k - x^*\|$")

    _mrkempty =  mlines.Line2D([], [], color='black', marker='', linestyle='None',label=r'$c$')
    _mrk1 = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                            markersize=10, label='0.1')
    _mrk2 = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                            markersize=10, label='0.08')
    _mrk3 = mlines.Line2D([], [], color='black', marker='+', linestyle='None',
                            markersize=10, label='0.05')
    _mrk4 = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                            markersize=10, label='0.02')
    _mrk5 = mlines.Line2D([], [], color='black', marker='*', linestyle='None',
                            markersize=10, label='0.01')

    for axfoo in [ax, ax2, ax3, ax4]:
        box = axfoo.get_position()
        axfoo.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        legend1 = axfoo.legend(handles=[_gdline, _nestline, _lmdline, _lamdline], bbox_to_anchor=(1.02, 0.7),
                                loc='center left', borderaxespad=0.)
        axfoo.legend(handles=[_mrkempty, _mrk1,_mrk2,_mrk3,_mrk4,_mrk5], bbox_to_anchor=(1.02,0.3),
                                loc='center left', borderaxespad=0.)
        axfoo.add_artist(legend1)
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
    fig, ax = plt.subplots(figsize = (8.0,4.8), dpi = 150) # loss plot
    fig2, ax2 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # fwdbwd plot
    fig3, ax3 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # log-loss plot 
    ax3.set_yscale('log')
    if LONGTIME:
        ax3.set_xscale('log')
        ax3.set_xlim(right = 10**3)
    ax3.plot(gdloss[:10**3]- true_min + eps, color='k', label = 'gd')
    ax3.plot(nesterovloss[:10**3]- true_min + eps, color='green', label = 'nesterov')
    fig4, ax4 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # l2 distance to minimum
    ax.axhline(true_min, label = "true min", linestyle = '--')
    min_attain = np.amin(np.concatenate((gdloss[:10**3], nesterovloss[:10**3])))

    # load standard
    # loss_mom = np.load(os.path.join(datpath, "maptransfer", "standard", "loss_mom.npy"))
    # loss_nomom = np.load(os.path.join(datpath, "maptransfer", "standard", "loss_nomom.npy"))
    # closeness_mom = np.load(os.path.join(datpath, "maptransfer", "standard", "fwdbwd_mom.npy"))
    # closeness_nomom = np.load(os.path.join(datpath, "maptransfer", "standard", "fwdbwd_nomom.npy"))
    # l2_mom = np.load(os.path.join(datpath, "maptransfer", "standard", "l2_mom.npy"))
    # l2_nomom = np.load(os.path.join(datpath, "maptransfer", "standard", "l2_nomom.npy"))


    loss_mom = np.load(os.path.join(datpath, "stepsizeext", "loss_mom_" + "recip" + ".npy"))
    loss_nomom = np.load(os.path.join(datpath, "stepsizeext", "loss_nomom_" + "recip" + ".npy"))
    closeness_mom = np.load(os.path.join(datpath, "stepsizeext", "fwdbwd_mom_" + "recip" + ".npy"))
    closeness_nomom = np.load(os.path.join(datpath, "stepsizeext", "fwdbwd_nomom_" + "recip" + ".npy"))
    l2_mom = np.load(os.path.join(datpath, "stepsizeext", "l2_mom_" + "recip" + ".npy"))
    l2_nomom = np.load(os.path.join(datpath, "stepsizeext", "l2_nomom_" + "recip" + ".npy"))
    min_attain = np.amin(np.concatenate(([min_attain], loss_mom, loss_nomom)))
    # define useful constants
    init_err = loss_mom[0]
    init_closeness_mom = closeness_mom[0]
    init_l2 = l2_mom[0]
    currminfwdbwd = np.amin(np.concatenate((closeness_mom, closeness_nomom)))



    # plot
    ax.plot(loss_mom,  markevery=100, linestyle = "dashed", color='red')
    ax.plot(loss_nomom[~np.isinf(loss_nomom)],  markevery=100, linestyle = "dotted", color='blue')

    ax2.plot(closeness_mom,  markevery=100, linestyle = "dashed", color='red')
    ax2.plot(closeness_nomom,  markevery=100, linestyle = "dotted", color='blue')

    ax3.loglog(loss_mom - true_min + eps,  markevery=0.1, linestyle = "dashed", color='red')
    ax3.loglog(loss_nomom[~np.isinf(loss_nomom)] - true_min + eps,  markevery=0.1, linestyle = "dotted", color='blue')

    ax4.plot(l2_mom,  markevery=100, linestyle = "dashed", color='red')
    ax4.plot(l2_nomom,  markevery=100, linestyle = "dotted", color='blue')


    # load switched
    loss_mom = np.load(os.path.join(datpath, "maptransfer", "switch", "loss_mom.npy"))
    loss_nomom = np.load(os.path.join(datpath, "maptransfer", "switch", "loss_nomom.npy"))
    closeness_mom = np.load(os.path.join(datpath, "maptransfer", "switch", "fwdbwd_mom.npy"))
    closeness_nomom = np.load(os.path.join(datpath, "maptransfer", "switch", "fwdbwd_nomom.npy"))
    l2_mom = np.load(os.path.join(datpath, "maptransfer", "switch", "l2_mom.npy"))
    l2_nomom = np.load(os.path.join(datpath, "maptransfer", "switch", "l2_nomom.npy"))
    min_attain = np.amin(np.concatenate(([min_attain], loss_mom, loss_nomom)))
    # plot
    ax.plot(loss_mom, markevery=100, linestyle = (0,(10,5)), color='orange')
    ax.plot(loss_nomom[~np.isinf(loss_nomom)],  markevery=100, linestyle = 'dashdot', color='m')

    ax2.plot(closeness_mom,  markevery=100, linestyle = (0,(10,5)), color='orange')
    ax2.plot(closeness_nomom,  markevery=100, linestyle = 'dashdot', color='m')

    ax3.loglog(loss_mom - true_min + eps,  markevery=0.1, linestyle = (0,(10,5)), color='orange')
    ax3.loglog(loss_nomom[~np.isinf(loss_nomom)] - true_min + eps,  markevery=0.1, linestyle = 'dashdot', color='m')

    ax4.plot(l2_mom,  markevery=100, linestyle = (0,(10,5)), color='orange')
    ax4.plot(l2_nomom,  markevery=100, linestyle = 'dashdot', color='m')


    # rest of plot
    # rest of plot
    if TITLE:
        fig.suptitle("Loss")
        fig2.suptitle("Fwdbwd error")
        fig3.suptitle("Loss minus minimum recon")
        fig4.suptitle("L2 distance to true min")

    ax.set_ylim([true_min.item()*0.95, init_err*1.5])
    ax2.set_ylim([currminfwdbwd.item()*0.95, init_closeness_mom*1.5])
    ax3.set_ylim([(min_attain-true_min)*0.9, (init_err-true_min)*2.5])
    ax4.set_ylim([0,init_l2*1.5])

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$f(x_k)$")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"$\|(\nabla M_\theta^* \circ \nabla M_\theta - I) (x_k)\|$")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel(r"$\|f(x_k) - f^*\|$")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel(r"$\|x_k - x^*\|$")

    _mrkempty =  mlines.Line2D([], [], color='black', marker='', linestyle='None',label=r'$c$')
    _mrk1 = mlines.Line2D([], [], color='red', marker='', linestyle='dashed', label='LAMD')
    _mrk2 = mlines.Line2D([], [], color='blue', marker='', linestyle='dotted', label='LMD')
    _mrk3 = mlines.Line2D([], [], color='orange', marker='', linestyle=(0,(7,2)), label='LAMD Transfer')
    _mrk4 = mlines.Line2D([], [], color='m', marker='', linestyle='dashdot', label='LMD Transfer')


    for axfoo in [ax, ax2, ax3, ax4]:
        box = axfoo.get_position()
        axfoo.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        axfoo.legend(handles=[_gdline, _nestline, _mrk1,_mrk2,_mrk3,_mrk4], bbox_to_anchor=(1.02,0.5),
                                loc='center left', borderaxespad=0.)
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
    fig, ax = plt.subplots(figsize = (8.0,4.8), dpi = 150) # loss plot
    fig2, ax2 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # fwdbwd plot
    
    true_min = np.load(os.path.join(datpath, "true_min_val_alttransform.npy"))
    gdloss = np.load(os.path.join(datpath, "true_gd_progression_alttransform.npy"))
    #gdloss_recip = np.load(os.path.join(datpath, "recip_gd_progression.npy"))
    nesterovloss = np.load(os.path.join(datpath, "nesterov_progression_alttransform.npy"))

    fig3, ax3 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # log-loss plot 
    ax3.set_yscale('log')
    if LONGTIME:
        ax3.set_xscale('log')
        ax3.set_xlim(right = MAXITERS)
    ax3.plot(gdloss[:MAXITERS]- true_min + eps, color='k', label = 'gd')
    ax3.plot(nesterovloss[:MAXITERS]- true_min + eps, color='green', label = 'nesterov')
    min_attain = np.amin(np.concatenate((gdloss[:MAXITERS], nesterovloss[:MAXITERS])))
    ax.axhline(true_min, label = "true min", linestyle = '--')
    fig4, ax4 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # l2 distance to minimum
    # TYPE 2: t_k = c/n
    for c, marker in zip([0.1,0.08,0.05,0.02,0.01], ['o','x','+', '^', '*']):
        loss_mom = np.load(os.path.join(datpath, "alttransform", "loss_mom_" + str(c*1000) + ".npy"))
        loss_nomom = np.load(os.path.join(datpath, "alttransform", "loss_nomom_" + str(c*1000) + ".npy"))
        closeness_mom = np.load(os.path.join(datpath, "alttransform", "fwdbwd_mom_" + str(c*1000) + ".npy"))
        closeness_nomom = np.load(os.path.join(datpath, "alttransform", "fwdbwd_nomom_" + str(c*1000) + ".npy"))
        l2_mom = np.load(os.path.join(datpath, "alttransform", "l2_mom_" + str(c*1000) + ".npy"))
        l2_nomom = np.load(os.path.join(datpath, "alttransform", "l2_nomom_" + str(c*1000) + ".npy"))
        init_err = loss_mom[0]
        init_l2 = l2_mom[0]
        ax.plot(loss_mom, label = str(c) + " mom", marker = marker, markevery=100, linestyle = "dashed", color='red')
        ax.plot(loss_nomom[~np.isinf(loss_nomom)], label = str(c) + " nomom", marker = marker, markevery=100, linestyle = "dotted", color='blue')

        ax2.plot(closeness_mom, label = str(c) + " mom", marker = marker, markevery=100, linestyle = "dashed", color='red')
        ax2.plot(closeness_nomom, label = str(c) + " nomom", marker = marker, markevery=100, linestyle = "dotted", color='blue')

        ax3.loglog(loss_mom - true_min + eps, label = str(c) + " mom", marker = marker, markevery=0.1, linestyle = "dashed", color='red')
        ax3.loglog(loss_nomom[~np.isinf(loss_nomom)] - true_min + eps, label = str(c) + " nomom", marker = marker, markevery=0.1, linestyle = "dotted", color='blue')

    
        min_attain = np.amin(np.concatenate(([min_attain], loss_mom, loss_nomom)))
        ax4.plot(l2_mom, label = str(c) + " mom", marker = marker, markevery=100, linestyle = "dashed", color='red')
        ax4.plot(l2_nomom, label = str(c) + " nomom", marker = marker, markevery=100, linestyle = "dotted", color='blue')


    if TITLE:
        fig.suptitle("Loss")
        fig2.suptitle("Fwdbwd error")
        fig3.suptitle("Loss minus minimum recon")
        fig4.suptitle("L2 distance to true min")

    ax.set_ylim([true_min*0.95, init_err*1.1])
    ax2.set_ylim([currminfwdbwd.item()*0.95, init_closeness_mom*1.5])
    ax3.set_ylim([(min_attain-true_min)*0.9, (init_err-true_min)*2.5])
    ax4.set_ylim([0,init_l2*1.5])

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$f(x_k)$")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"$\|(\nabla M_\theta^* \circ \nabla M_\theta - I) (x_k)\|$")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel(r"$\|f(x_k) - f^*\|$")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel(r"$\|x_k - x^*\|$")

    _mrkempty =  mlines.Line2D([], [], color='black', marker='', linestyle='None',label=r'$c$')
    _mrk1 = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                            markersize=10, label='0.1')
    _mrk2 = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                            markersize=10, label='0.08')
    _mrk3 = mlines.Line2D([], [], color='black', marker='+', linestyle='None',
                            markersize=10, label='0.05')
    _mrk4 = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                            markersize=10, label='0.02')
    _mrk5 = mlines.Line2D([], [], color='black', marker='*', linestyle='None',
                            markersize=10, label='0.01')

    for axfoo in [ax, ax2, ax3, ax4]:
        box = axfoo.get_position()
        axfoo.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        legend1 = axfoo.legend(handles=[_gdline, _nestline, _lmdline, _lamdline], bbox_to_anchor=(1.02, 0.7),
                                loc='center left', borderaxespad=0.)
        axfoo.legend(handles=[_mrkempty, _mrk1,_mrk2,_mrk3,_mrk4,_mrk5], bbox_to_anchor=(1.02,0.3),
                                loc='center left', borderaxespad=0.)
        axfoo.add_artist(legend1)

    if PDF:
        fig.savefig(os.path.join(figs_dir, "alttrans_losses.pdf"))
        fig2.savefig(os.path.join(figs_dir, "alttrans_fwdbwd.pdf"))
        fig3.savefig(os.path.join(figs_dir, "alttrans_loglosses.pdf"))
        fig4.savefig(os.path.join(figs_dir, "alttrans_l2totrue.pdf"))
    if PNG:
        fig.savefig(os.path.join(figs_dir, "alttrans_losses"))
        fig2.savefig(os.path.join(figs_dir, "alttrans_fwdbwd"))
        fig3.savefig(os.path.join(figs_dir, "alttrans_loglosses"))
        fig4.savefig(os.path.join(figs_dir, "alttrans_l2totrue"))
    # fig4.savefig(os.path.join(figs_dir, "alttrans_l2totrue.pdf"))
        # save

    end = time.time()
    print("alt ray transform, elapsed time", end-start)
    

if DECONV:
    MAXITERS = 1000
    fig, ax = plt.subplots(figsize = (8.0,4.8), dpi = 150) # loss plot
    fig2, ax2 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # fwdbwd plot
    fig4, ax4 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # l2 distance to minimum
    # 
    true_min = np.load(os.path.join(datpath, "true_min_val_deconv.npy"))
    gdloss = np.load(os.path.join(datpath, "true_gd_progression_deconv.npy"))
    #gdloss_recip = np.load(os.path.join(datpath, "recip_gd_progression.npy"))
    nesterovloss = np.load(os.path.join(datpath, "nesterov_progression_deconv.npy"))

    fig3, ax3 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # log-loss plot 
    ax3.set_yscale('log')
    if LONGTIME:
        ax3.set_xscale('log')
        ax3.set_xlim(right = MAXITERS)
    true_min = np.amin(np.concatenate((gdloss, nesterovloss))) - 1e-6
    ax3.plot(gdloss[:MAXITERS]- true_min + eps, color='k', label = 'gd')
    ax3.plot(nesterovloss[:MAXITERS]- true_min + eps, color='green', label = 'nesterov')
    min_attain = np.amin(np.concatenate((gdloss[:MAXITERS], nesterovloss[:MAXITERS])))
    
    ax.axhline(true_min, label = "true min", linestyle = '--')


    
    # TYPE 2: t_k = c/n
    for c in [0.1,0.08,0.05,0.02,0.01]:
        loss_mom = np.load(os.path.join(datpath, "deconv", "loss_mom_" + str(c*1000) + ".npy"))
        loss_nomom = np.load(os.path.join(datpath, "deconv", "loss_nomom_" + str(c*1000) + ".npy"))
        closeness_mom = np.load(os.path.join(datpath, "deconv", "fwdbwd_mom_" + str(c*1000) + ".npy"))
        closeness_nomom = np.load(os.path.join(datpath, "deconv", "fwdbwd_nomom_" + str(c*1000) + ".npy"))
        l2_mom = np.load(os.path.join(datpath, "deconv", "l2_mom_" + str(c*1000) + ".npy"))
        l2_nomom = np.load(os.path.join(datpath, "deconv", "l2_nomom_" + str(c*1000) + ".npy"))
        init_err = loss_mom[0]
        init_l2 = l2_mom[0]
        ax.plot(loss_mom, label = str(c) + " mom", marker = 'o')
        ax.plot(loss_nomom[~np.isinf(loss_nomom)], label = str(c) + " nomom", marker = 'x')

        ax2.plot(closeness_mom, label = str(c) + " mom", marker = 'o')
        ax2.plot(closeness_nomom, label = str(c) + " nomom", marker = 'x')

        ax3.plot(loss_mom - true_min + eps, label = str(c) + " mom", marker = 'o')
        ax3.plot(loss_nomom[~np.isinf(loss_nomom)] - true_min + eps, label = str(c) + " nomom", marker = 'x')
    
        ax4.plot(l2_mom, label = str(c) + " mom", marker = 'o')
        ax4.plot(l2_nomom, label = str(c) + " nomom", marker = 'x')

        min_attain = np.amin(np.concatenate(([min_attain], loss_mom, loss_nomom)))

        
    fig.suptitle("Loss")
    fig2.suptitle("Fwdbwd error")
    fig3.suptitle("Loss minus minimum recon")
    fig4.suptitle("L2 distance to true min")

    ax.set_ylim([true_min*0.95, init_err*1.1])
    ax2.set_ylim([currminfwdbwd.item()*0.95, init_closeness_mom*1.5])
    ax3.set_ylim([(min_attain-true_min)*0.9, (init_err-true_min)*2.5])
    ax4.set_ylim([0,init_l2*1.5])

    fig.legend()
    fig2.legend() 
    fig3.legend()
    fig4.legend()

    if PDF:
        fig.savefig(os.path.join(figs_dir, "deconv_losses.pdf"))
        fig2.savefig(os.path.join(figs_dir, "deconv_fwdbwd.pdf"))
        fig3.savefig(os.path.join(figs_dir, "deconv_loglosses.pdf"))
        fig4.savefig(os.path.join(figs_dir, "deconv_l2totrue.pdf"))
    if PNG:
        fig.savefig(os.path.join(figs_dir, "deconv_losses"))
        fig2.savefig(os.path.join(figs_dir, "deconv_fwdbwd"))
        fig3.savefig(os.path.join(figs_dir, "deconv_loglosses"))
        fig4.savefig(os.path.join(figs_dir, "deconv_l2totrue"))

    end = time.time()
    print("deconv, elapsed time", end-start)
# %%
