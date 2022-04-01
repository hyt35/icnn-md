# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:01:37 2022

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
#import parse_import
import logging
from datetime import datetime
import torch.nn.functional as F
import time
device = 'cuda'

#logging.basicConfig(filename=datetime.now().strftime('logs/%d-%m_%H:%M-lowiter-postpad.log'), filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
#logger = logging.getLogger()

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
        self.wz = nn.ModuleList([nn.Conv2d(self.n_filters, self.n_filters, self.kernel_size, stride=1, padding=self.padding, bias=False)\
                                 for i in range(self.n_layers)])
        
        #these layers can have arbitrary weights
        self.wx_quad = nn.ModuleList([nn.Conv2d(self.n_in_channels, self.n_filters, self.kernel_size, stride=1, padding=self.padding, bias=False)\
                                 for i in range(self.n_layers+1)])
    
        self.wx_lin = nn.ModuleList([nn.Conv2d(self.n_in_channels, self.n_filters, self.kernel_size, stride=1, padding=self.padding, bias=True)\
                                 for i in range(self.n_layers+1)])
        
        #one final conv layer with nonnegative weights
        self.final_conv2d = nn.Conv2d(self.n_filters, self.n_in_channels, self.kernel_size, stride=1, padding=self.padding, bias=False)
        
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

laplacian_weight = torch.Tensor([[[[0,-1,0],
                    [-1,4,-1],
                    [0,-1,0]]]]).repeat(3,1,1,1).to(device)

padder = nn.ReplicationPad2d(1)

noise_level = 0.05
reg_param = 0.3
stepsize = 0.2


# Initialize models
# this one is poor
# strong_convexity = 0.1
# icnn_fwd = ICNN(num_in_channels=3, num_filters=50, kernel_dim=3, num_layers=3, strong_convexity = strong_convexity).to(device)
# icnn_fwd.initialize_weights()
# icnn_bwd = ICNN(num_in_channels=3, num_filters=75, kernel_dim=3, num_layers=5, strong_convexity = 1).to(device)

# icnn_fwd.load_state_dict(torch.load('checkpoints/lowiter_postpad/1450fwd_small', map_location='cuda:0'))
# icnn_bwd.load_state_dict(torch.load('checkpoints/lowiter_postpad/1450bwd_small', map_location='cuda:0'))


# this one works!!
# In line with training
strong_convexity = 0.1
icnn_fwd = ICNN(num_in_channels=3, num_filters=50, kernel_dim=3, num_layers=3, strong_convexity = strong_convexity).to(device)
icnn_fwd.initialize_weights()
icnn_bwd = ICNN(num_in_channels=3, num_filters=75, kernel_dim=3, num_layers=5, strong_convexity = 0.5).to(device)

icnn_fwd.load_state_dict(torch.load('checkpoints/stackedloss/1500fwd_small', map_location='cuda:0'))
icnn_bwd.load_state_dict(torch.load('checkpoints/stackedloss/1500bwd_small', map_location='cuda:0'))


icnn_fwd.eval()
icnn_bwd.eval()
# MODIFY THIS PATH TO YOUR DATA
stl10_data_test = torchvision.datasets.STL10('./stl10', split='test', transform=torchvision.transforms.ToTensor(), folds=1)
test_dataloader = torch.utils.data.DataLoader(stl10_data_test, batch_size=1)

#%%
# This one plots the MD iterations
# in order: original, noisy, fwd(noisy), bwd(fwd(noisy)), 20x MD iterations, result after 20x GD
ctr = 0

for batch_, _ in test_dataloader:
    if ctr != 0:
        ctr-=1
        continue
    
    #batch = batch_.to(device)
    batch_noisy = (batch_ + noise_level * torch.randn_like(batch_, requires_grad = True)).to(device)
    
    batch_noisy_ = batch_noisy.cpu().detach().numpy()
    
    
    plt.figure()
    plt.imshow(batch_[0,:,:,:].permute(1,2,0))
    plt.figure()
    plt.imshow(batch_noisy_[0,:,:,:].transpose(1,2,0))
    
    def recon_err(img):
        #bs_img, c_img, h_img, w_img = img.size()
        tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
        tv = (tv_h+tv_w)
        fidelity = torch.pow(img-batch_noisy,2).sum()
        return (fidelity + reg_param*tv)/2
    
    def recon_err_grad(img):
        # laplacian = 4*img[:,:,1:-1,1:-1] - img[:,:,:-2,1:-1] - img[:,:,2:,1:-1] \
        #                                 - img[:,:,1:-1,2:]-img[:,:,1:-1,:-2]
        # laplacian = padder(laplacian)
        #laplacian = padder(F.conv2d(img, laplacian_weight, groups=3))
        #return img - batch_noisy + reg_param*laplacian
        return autograd.grad(recon_err(img),img)[0]
    
    fwd = batch_noisy
    
    closeness = torch.linalg.vector_norm(icnn_bwd(icnn_fwd(batch_noisy))-batch_noisy)
    print("closeness", closeness.item())
    del closeness
    
    foo = icnn_fwd(fwd)
    #foo = iresnet_model.inverse(fwd)
    plt.figure()
    plt.imshow(foo.cpu().detach().clone()[0,:,:,:].permute(1,2,0))
    #del foo
    bar = icnn_bwd(foo)
    plt.figure()
    plt.imshow(bar.cpu().detach().clone()[0,:,:,:].permute(1,2,0))
    del bar
    

    torch.cuda.empty_cache()

    
    print("Initial recon err", recon_err(fwd).item())
    ## MIRROR DESCENT

    for i in range(20):
        fwd = fwd.clamp(0,1)
        fwd.requires_grad_()
        fwdGrad0 = recon_err_grad(fwd).detach().clone()
        fwd = icnn_bwd(icnn_fwd(fwd) - stepsize*fwdGrad0)
        #fwd = icnn_bwd(icnn_fwd(fwd))
        fwd_ = fwd.cpu().detach().numpy()
        plt.figure()
        plt.imshow(fwd_[0,:,:,:].transpose(1,2,0))
        print("MD recon", recon_err(fwd).item())
    
    fwd = batch_noisy
    ## GRADIENT DESCENT
    
    for i in range(20):
        fwd.requires_grad_()
        #fwdGrad0 = autograd.grad(recon_err(fwd), fwd)[0].detach()
        fwdGrad0 = recon_err_grad(fwd)
        #fwd = iresnet_model.inverse(icnn(fwd) - stepsize*fwdGrad0) + icnn(fwd) - stepsize*fwdGrad0
        fwd = fwd - stepsize*fwdGrad0
        #fwd = icnn(fwd)
        print("GD recon", recon_err(fwd).item())
        #fwd = (fwd-fwd.min())/(fwd.max()-fwd.min())

    fwd_ = fwd.cpu().detach().numpy()
    fwd_ = (fwd_-fwd_.min())/(fwd_.max()-fwd_.min())
    plt.figure()
    plt.imshow(fwd_[0,:,:,:].transpose(1,2,0))

    break
#%%
# Plot pointwise ratio of fwd map to noisy image (per channel)
plt.figure()
plt.imshow(torch.clamp(foo/batch_noisy,-1,1).detach().cpu().numpy()[0,0])
plt.colorbar()
plt.figure()
plt.imshow(torch.clamp(foo/batch_noisy,-1,1).detach().cpu().numpy()[0,1])
plt.colorbar()
plt.figure()
plt.imshow(torch.clamp(foo/batch_noisy,-1,1).detach().cpu().numpy()[0,2])
plt.colorbar()
#%%
# This one plots the evolution of loss
ctr = 0
stepsize_scales = [0,1/8,1/4,1/2,1]
#stepsize_scales = [0,0.1,0.2,0.3,0.5,0.6,0.7,0.8,0.9,1]
noise_level = 0.05

for batch_, _ in test_dataloader:
    if ctr != 0:
        ctr-=1
        continue
    
    #batch = batch_.to(device)
    batch_noisy = (batch_ + noise_level * torch.randn_like(batch_, requires_grad = True)).to(device)
    
    batch_noisy_ = batch_noisy.cpu().detach().numpy()
    plt.figure()
    plt.imshow(batch_[0,:,:,:].permute(1,2,0))

    
    def recon_err(img):
        #bs_img, c_img, h_img, w_img = img.size()
        tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
        tv = (tv_h+tv_w)
        fidelity = torch.pow(img-batch_noisy,2).sum()
        return (fidelity + reg_param*tv)/2
    
    def recon_err_grad(img):
        # laplacian = 4*img[:,:,1:-1,1:-1] - img[:,:,:-2,1:-1] - img[:,:,2:,1:-1] \
        #                                 - img[:,:,1:-1,2:]-img[:,:,1:-1,:-2]
        # laplacian = padder(laplacian)
        #laplacian = padder(F.conv2d(img, laplacian_weight, groups=3))
        #return img - batch_noisy + reg_param*laplacian
        return autograd.grad(recon_err(img),img)[0]
    
    fwd = batch_noisy
    
    closeness = torch.linalg.vector_norm(icnn_bwd(icnn_fwd(batch_noisy))-batch_noisy)
    print("closeness", closeness.item())
    del closeness
    
    
    initloss = recon_err(batch_noisy).item()
    
    plt.figure(figsize=(12,10))
    for stepsize_scale in stepsize_scales:
        
        fwd = batch_noisy
        initloss = recon_err(batch_noisy).item()
        torch.cuda.empty_cache()
        mdloss = [initloss]
        gdloss = [initloss]
        
        print("Initial recon err", initloss)
        ## MIRROR DESCENT
        
        start = time.time()
        for i in range(20):
            #fwd = fwd.clamp(0,1)
            fwd.requires_grad_()
            fwdGrad0 = recon_err_grad(fwd).detach().clone()
            fwd = icnn_bwd(icnn_fwd(fwd) - stepsize*fwdGrad0*stepsize_scale)
            fwd_ = fwd.cpu().detach().numpy()
            # plt.figure()
            # plt.imshow(fwd_[0,:,:,:].transpose(1,2,0))
            #print("MD recon", recon_err(fwd).item())
            mdloss.append(recon_err(fwd).item())
        end = time.time(); elapsed = end-start; print("md time", elapsed)
        fwd = batch_noisy
        ## GRADIENT DESCENT
        
        start = time.time()
        for i in range(20):
            fwd.requires_grad_()
            #fwdGrad0 = autograd.grad(recon_err(fwd), fwd)[0].detach()
            fwdGrad0 = recon_err_grad(fwd)
            #fwd = iresnet_model.inverse(icnn(fwd) - stepsize*fwdGrad0) + icnn(fwd) - stepsize*fwdGrad0
            fwd = fwd - stepsize*fwdGrad0*stepsize_scale
            #fwd = icnn(fwd)
            #print("GD recon", recon_err(fwd).item())
            gdloss.append(recon_err(fwd).item())
            #fwd = (fwd-fwd.min())/(fwd.max()-fwd.min())
        end = time.time(); elapsed = end-start; print("gd time", elapsed)
        ## ADAM
        adamloss = []
        
        par = batch_noisy.clone().detach()
        par.requires_grad_()
        # lr here is smaller: 0.05 instead of 0.2
        optimizer = torch.optim.Adam([par], lr=stepsize_scale*0.05)
        start = time.time()
        for i in range(21):
            optimizer.zero_grad()
            loss = recon_err(par)
            loss.backward(retain_graph=True)
            optimizer.step()
            
            #print("adam recon", recon_err(par).item())
            adamloss.append(loss.item())
            #fwd = (fwd-fwd.min())/(fwd.max()-fwd.min())
        end = time.time(); elapsed = end-start; print("adam time", elapsed)
        
        
        plt.plot(mdloss, label = "mdloss " + str(stepsize_scale), marker='x')
        plt.plot(gdloss, label = "gdloss " + str(stepsize_scale), marker='o')
        plt.plot(adamloss, label = "adamloss " + str(stepsize_scale), marker='v')
        # fwd_ = fwd.cpu().detach().numpy()
        # #fwd_ = (fwd_-fwd_.min())/(fwd_.max()-fwd_.min())
        # plt.figure()
        # plt.imshow(fwd_[0,:,:,:].transpose(1,2,0))
        plt.legend()
    break

