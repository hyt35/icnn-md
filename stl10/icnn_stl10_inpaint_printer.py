# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:01:37 2022

@author: hongy
"""
#from icnn import DenseICGN
#from denoising_nets_for_mcmc import ICNN
import numpy as np
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


device = 'cuda'
checkpoint_path = '/local/scratch/public/hyt35/ICNN-MD/ICNN-STL10/checkpoints/Apr20_inpainting/780'

stl10_data = torchvision.datasets.STL10('/local/scratch/public/hyt35/datasets/STL10', split='test', transform=torchvision.transforms.ToTensor(), folds=1, download=True)

torch.cuda.set_device(1)

mask_ = torch.load('mask_30.pt') # mask with 70% remove
#mask = torch.load('mask_40.pt') # with 60% remove
mask = mask_.to(device)

def compute_gradient(net, inp):
    inp_with_grad = inp.requires_grad_(True)
    out = net(inp_with_grad)  
    fake = torch.cuda.FloatTensor(np.ones(out.shape)).requires_grad_(False)
    # Get gradient w.r.t. input
    gradients = autograd.grad(outputs=out, inputs=inp_with_grad,
                              grad_outputs=fake, create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    return gradients

class ICNN(nn.Module):
    def __init__(self, num_in_channels=1, num_filters=64, kernel_dim=5, num_layers=10, strong_convexity = 0.5):
        super(ICNN, self).__init__()
        self.n_in_channels = num_in_channels
        self.n_layers = num_layers
        self.n_filters = num_filters
        self.kernel_size = kernel_dim
        self.padding = (self.kernel_size-1)//2
        #these layers should have non-negative weights
        self.wz = nn.ModuleList([nn.Conv2d(self.n_filters, self.n_filters, self.kernel_size, stride=1, padding=self.padding, padding_mode='circular', bias=False)\
                                 for i in range(self.n_layers)])
        
        #these layers can have arbitrary weights
        self.wx_quad = nn.ModuleList([nn.Conv2d(self.n_in_channels, self.n_filters, self.kernel_size, stride=1, padding=self.padding, padding_mode='circular', bias=False)\
                                 for i in range(self.n_layers+1)])
    
        self.wx_lin = nn.ModuleList([nn.Conv2d(self.n_in_channels, self.n_filters, self.kernel_size, stride=1, padding=self.padding, padding_mode='circular', bias=True)\
                                 for i in range(self.n_layers+1)])
        
        #one final conv layer with nonnegative weights
        self.final_conv2d = nn.Conv2d(self.n_filters, self.n_in_channels, self.kernel_size, stride=1, padding=self.padding, padding_mode='circular', bias=False)
        
        #slope of leaky-relu
        self.negative_slope = 0.2 
        self.strong_convexity = strong_convexity
        
        
    def scalar(self, x):
        z = torch.nn.functional.leaky_relu(self.wx_quad[0](x)**2 + self.wx_lin[0](x), negative_slope=self.negative_slope)
        for layer in range(self.n_layers):
            z = torch.nn.functional.leaky_relu(self.wz[layer](z) + self.wx_quad[layer+1](x)**2 + self.wx_lin[layer+1](x), negative_slope=self.negative_slope)
        z = self.final_conv2d(z)
        z_avg = torch.nn.functional.avg_pool2d(z, z.size()[2:]).view(z.size()[0], -1)
        
        return z_avg# + .5 * self.strong_convexity * (x ** 2).sum(dim=[1,2,3]).reshape(-1, 1)
    
    def forward(self, x):
        foo = compute_gradient(self.scalar, x)
        return (1-self.strong_convexity)*foo + self.strong_convexity*x
    
    #a weight initialization routine for the ICNN
    def initialize_weights(self, min_val=0.0, max_val=0.001, device=device):
        for layer in range(self.n_layers):
            self.wz[layer].weight.data = min_val + (max_val - min_val)\
            * torch.rand(self.n_filters, self.n_filters, self.kernel_size, self.kernel_size).to(device)
        
        self.final_conv2d.weight.data = min_val + (max_val - min_val)\
        * torch.rand(1, self.n_filters, self.kernel_size, self.kernel_size).to(device)
        return self
    
    #a zero clipping functionality for the ICNN (set negative weights to 0)
    def zero_clip_weights(self): 
        for layer in range(self.n_layers):
            self.wz[layer].weight.data.clamp_(0)
        
        self.final_conv2d.weight.data.clamp_(0)
        return self 

class ICNNCouple(nn.Module):
    def __init__(self, stepsize_init = 0.01, num_iters = 10, stepsize_clamp = (0.001,0.1)):
        super(ICNNCouple, self).__init__()
        self.fwd_model = None
        self.bwd_model = None
        self.stepsize = nn.Parameter(stepsize_init * torch.ones(num_iters).to(device))
        self.num_iters = num_iters
        self.ssmin = stepsize_clamp[0]
        self.ssmax = stepsize_clamp[1]
        # Logger
        # Checkpoint
        
    def init_fwd(self, num_in_channels=1, num_filters=64, kernel_dim=5, num_layers=10, strong_convexity = 0.5):
        self.fwd_model = ICNN(num_in_channels, num_filters, kernel_dim, num_layers, strong_convexity).to(device)
        return self
        
    def init_bwd(self, num_in_channels=1, num_filters=64, kernel_dim=5, num_layers=10, strong_convexity = 0.5):
        self.bwd_model = ICNN(num_in_channels, num_filters, kernel_dim, num_layers, strong_convexity).to(device)
        return self
    
    def clip_fwd(self):
        self.fwd_model.zero_clip_weights()
        return self
        
    def clip_bwd(self):
        self.bwd_model.zero_clip_weights()
        return self
        
    def forward(self, x, gradFun = None):
        if gradFun is None:
            raise RuntimeError("Gradient function not provided")
        with torch.no_grad():
            fwd = x
            for ss in self.stepsize:
                fwd = self.bwd_model(self.fwd_model(x) - ss*gradFun(fwd)) 
        return fwd
    
    def clamp_stepsizes(self):
        with torch.no_grad():
            self.stepsize.clamp_(self.ssmin,self.ssmax)
        return self
            
    def fwdbwdloss(self, x):
        return torch.linalg.vector_norm(self.bwd_model(self.fwd_model(x))-x, ord=1)


#%%
# Initialize models

icnn_couple = ICNNCouple(stepsize_init = 0.01, num_iters = 10, stepsize_clamp = (0.001,0.1))
icnn_couple.init_fwd(num_in_channels=3, num_filters=60, kernel_dim=3, num_layers=3, strong_convexity = 0.1)
icnn_couple.fwd_model.initialize_weights()
icnn_couple.init_bwd(num_in_channels=3, num_filters=75, kernel_dim=3, num_layers=5, strong_convexity = 0.5)
icnn_couple.load_state_dict(torch.load(checkpoint_path, map_location='cuda:1'))

noise_level = 0.05
reg_param = 0.3
stepsize = 0.01
closeness_reg = 1.0
closeness_update_nepochs = 50
closeness_update_multi = 1.05
#%%
ctr = 0
stepsize_scales = [1/4,1/2,1,2,4]

if __name__ == '__main__': 

    icnn_couple.train()
    opt = torch.optim.Adam(icnn_couple.parameters(),lr=1e-5,betas=(0.9,0.99))
    test_dataloader = torch.utils.data.DataLoader(stl10_data, batch_size=1) # When training

    for idx, (batch_, _) in enumerate(test_dataloader):
        if ctr != 0:
            ctr-=1
            continue
        # add gaussian noise
        batch_masked_ = batch_*mask_
        batch_masked = batch_masked_.to(device)

        # define objective functions
        def recon_err(img):
            tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
            tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
            tv = (tv_h+tv_w)
            fidelity = torch.pow(img*mask-batch_masked,2).sum()
            return (fidelity + reg_param*tv)/2
        
        def recon_err_grad(img):
            return autograd.grad(recon_err(img), img)[0]

        initloss = recon_err(batch_masked).item()

        
        mdloss = [initloss]
        

        fwd = batch_masked.requires_grad_()

        for ss in icnn_couple.stepsize:
            #fwd = fwd.clamp(0,1)
            fwd.requires_grad_()
            fwdGrad0 = recon_err_grad(fwd).detach().clone()
            fwd = icnn_couple.bwd_model(icnn_couple.fwd_model(fwd) - ss*fwdGrad0)
            #fwd = icnn_bwd(icnn_fwd(fwd))
        fwd_ = fwd.cpu().detach().numpy()

        plt.imshow(fwd_[0].transpose(1,2,0))
        plt.savefig("figs/md_adaptive")


        max_mdloss = max(mdloss)
        min_mdloss = min(mdloss)

        for stepsize_scale in stepsize_scales:
            torch.cuda.empty_cache()
            fwd = batch_masked.requires_grad_()

            mdloss = [initloss]
            ## MD Fixed stepsize
            for i in range(20):
                fwd.requires_grad_()
                fwdGrad0 = recon_err_grad(fwd).detach().clone()
                fwd = icnn_couple.bwd_model(icnn_couple.fwd_model(fwd) - stepsize*stepsize_scale*fwdGrad0).detach()
                #fwd = icnn_bwd(icnn_fwd(fwd))
            fwd_ = fwd.cpu().detach().numpy()

            plt.imshow(fwd_[0].transpose(1,2,0))
            plt.savefig("figs/md_"+str(int(stepsize_scale*100)))


            gdloss = [initloss]
            ## GD
            fwd = batch_masked.requires_grad_()
            for i in range(20):
                fwd.requires_grad_()
                fwdGrad0 = recon_err_grad(fwd)
                fwd = fwd - stepsize*fwdGrad0*stepsize_scale
            plt.imshow(fwd_[0].transpose(1,2,0))
            plt.savefig("figs/gd_"+str(int(stepsize_scale*100)))

            ## ADAM
            adamloss = []
            tmp =  batch_masked.requires_grad_()
            par = tmp.clone().detach()
            par.requires_grad_()
            optimizer = torch.optim.Adam([par], lr=stepsize_scale*0.05)
            for i in range(21):
                optimizer.zero_grad()
                loss = recon_err(par)
                loss.backward(retain_graph=True)
                optimizer.step()

            fwd_ = par.cpu().detach().numpy()

            plt.imshow(fwd_[0].transpose(1,2,0))
            plt.savefig("figs/adam_"+str(int(stepsize_scale*100)))

        plt.imshow(batch_.numpy()[0].transpose(1,2,0))
        plt.savefig("figs/true")

        plt.imshow(batch_masked_.numpy()[0].transpose(1,2,0))
        plt.savefig("figs/masked")
        break

