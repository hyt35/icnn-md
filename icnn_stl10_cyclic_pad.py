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

logging.basicConfig(filename=datetime.now().strftime('logs/%d-%m_%H:%M-cyclicpad.log'), filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger()
stl10_data = torchvision.datasets.STL10('/local/scratch/public/hyt35/datasets/STL10', split='train', transform=torchvision.transforms.ToTensor(), folds=1, download=True)
args=parse_import.parse_commandline_args()
torch.cuda.set_device(2)


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


# laplacian_weight = torch.Tensor([[[[0,-1,0],
#                     [-1,4,-1],
#                     [0,-1,0]]]]).repeat(3,1,1,1).to(device)

#padder = nn.ReplicationPad2d(1)
#%%
# Initialize models

strong_convexity = 0.1
icnn_fwd = ICNN(num_in_channels=3, num_filters=50, kernel_dim=3, num_layers=3, strong_convexity = strong_convexity).to(device)
icnn_fwd.initialize_weights()
icnn_bwd = ICNN(num_in_channels=3, num_filters=75, kernel_dim=3, num_layers=5, strong_convexity = 0.5).to(device)


#%%
logger.info("fwd params"+str( sum(p.numel() for p in icnn_fwd.parameters())))
logger.info("bwd params"+str( sum(p.numel() for p in icnn_bwd.parameters())))
#%%
if args.from_checkpoint is not None:
    icnn_fwd.load_state_dict(torch.load(args.from_checkpoint+'fwd_small'))
    icnn_bwd.load_state_dict(torch.load(args.from_checkpoint+'bwd_small'))

n_epochs = args.num_epochs
noise_level = 0.05
reg_param = 0.3
stepsize = 0.01
bsize=10
#%%

if __name__ == '__main__': 
    if args.train:
        icnn_fwd.train()
        icnn_bwd.train()
        opt_fwd = torch.optim.Adam(icnn_fwd.parameters(),lr=1e-5,betas=(0.9,0.99))
        opt_bwd = torch.optim.Adam(icnn_bwd.parameters(),lr=1e-5,betas=(0.9,0.99))
        train_dataloader = torch.utils.data.DataLoader(stl10_data, batch_size=bsize) # When training

        for epoch in range(n_epochs):
            total_loss = 0
            total_fwdbwd = 0
            batch_loss = 0
            batch_fwdbwd = 0
            for idx, (batch_, _) in enumerate(train_dataloader):
                # add gaussian noise
                #batch = batch_.to(device)
                #batch_noisy = batch + noise_level * torch.randn_like(batch)
                batch_noisy_ = batch_ + noise_level * torch.randn_like(batch_)
                batch_noisy = batch_noisy_.to(device)
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
                    # laplacian = F.conv2d(img, laplacian_weight, groups=3,padding='same')
                    # return img - batch_noisy + reg_param*laplacian
                    return autograd.grad(recon_err(img), img)[0]

                fwd = batch_noisy.requires_grad_()
                loss = 0
                for i in range(10):
                    fwdGrad = recon_err_grad(fwd)
                    fwd = icnn_bwd(icnn_fwd(fwd) - stepsize*fwdGrad) 
                    loss += recon_err(fwd)
                #loss = recon_err(fwd)
                closeness = torch.linalg.vector_norm(icnn_bwd(icnn_fwd(batch_noisy)) - batch_noisy, ord=1)
                
                err = loss+closeness
                opt_fwd.zero_grad()
                opt_bwd.zero_grad()
                err.backward()
                opt_fwd.step()
                opt_bwd.step()
                icnn_fwd.zero_clip_weights()
                #icnn_bwd.zero_clip_weights()
                
                total_loss += err.item()
                total_fwdbwd += closeness.item()
                batch_loss += err.item()
                batch_fwdbwd += closeness.item()
                if(idx % args.num_batches == args.num_batches-1):
                    avg_loss = batch_loss/args.num_batches/bsize
                    avg_fwdbwd = batch_fwdbwd/args.num_batches/bsize
                    print("loss", loss.item(), "closeness", closeness.item())
                    train_log = "epoch:[{}/{}] batch:[{}/{}], avg_loss = {:.4f}, avg_fwdbwd = {:.4f}".\
                      format(epoch+1, args.num_epochs, idx+1, len(train_dataloader), avg_loss, avg_fwdbwd)
                    print(train_log)
                    logger.info(train_log)
                    batch_loss = 0
                    batch_fwdbwd = 0
                
            print("Epoch", epoch, "total loss", total_loss, "fwdbwd", total_fwdbwd)
            # Checkpoint
            if (epoch%args.checkpoint_freq == args.checkpoint_freq-1):
                torch.save(icnn_fwd.state_dict(), '/local/scratch/public/hyt35/ICNN-MD/ICNN-STL10/checkpoints/Apr6_cyclic_pad/'+str(epoch+1)+'fwd_small')
                torch.save(icnn_bwd.state_dict(), '/local/scratch/public/hyt35/ICNN-MD/ICNN-STL10/checkpoints/Apr6_cyclic_pad/'+str(epoch+1)+'bwd_small')
            # Log
                logger.info("\n====epoch:[{}/{}], epoch_loss = {:.2f}, epoch_fwdbwd = {:.4f}====\n".format(epoch+1, args.num_epochs, total_loss, total_fwdbwd))

        #%%
    # else:
    #     #%%
    #     icnn_fwd.eval()
    #     icnn_bwd.eval()
    #     stl10_data_test = torchvision.datasets.STL10('./stl10', split='train', transform=torchvision.transforms.ToTensor(), folds=1)
    #     test_dataloader = torch.utils.data.DataLoader(stl10_data_test, batch_size=1)
    #     for batch_, _ in test_dataloader:
    #         batch = batch_.to(device)
    #         batch_noisy = batch + noise_level * torch.randn_like(batch, requires_grad = True)
            
    #         batch_noisy_ = batch_noisy.cpu().detach().numpy()
    #         plt.figure()
    #         plt.imshow(batch_[0,:,:,:].permute(1,2,0))
    #         plt.figure()
    #         plt.imshow(batch_noisy_[0,:,:,:].transpose(1,2,0))
            
    #         def recon_err(img):
    #             #bs_img, c_img, h_img, w_img = img.size()
    #             tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
    #             tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
    #             tv = (tv_h+tv_w)
    #             fidelity = torch.pow(img-batch_noisy,2).sum()
    #             return (fidelity + reg_param*tv)/2
            
    #         def recon_err_grad(img):
    #             # laplacian = 4*img[:,:,1:-1,1:-1] - img[:,:,:-2,1:-1] - img[:,:,2:,1:-1] \
    #             #                                 - img[:,:,1:-1,2:]-img[:,:,1:-1,:-2]
    #             # laplacian = padder(laplacian)
    #             laplacian = F.conv2d(img, laplacian_weight, groups=3,padding='same')
    #             return img - batch_noisy + reg_param*laplacian
            
    #         fwd = batch_noisy
            
    #         closeness = torch.linalg.vector_norm(icnn_bwd(icnn_fwd(batch_noisy))-batch_noisy)
    #         print(closeness)
            
    #         foo = icnn_fwd(fwd)
    #         #foo = iresnet_model.inverse(fwd)
    #         plt.figure()
    #         plt.imshow(foo.cpu().detach().clone()[0,:,:,:].permute(1,2,0))
    #         #del foo
    #         bar = icnn_bwd(foo)
    #         plt.figure()
    #         plt.imshow(bar.cpu().detach().clone()[0,:,:,:].permute(1,2,0))
    #         #del foo

    #         torch.cuda.empty_cache()
    #         print("Initial recon err", recon_err(fwd).item())
    #         ## MIRROR DESCENT
    #         for i in range(25):
    #             fwd = fwd.clamp(0,1)
    #             fwd.requires_grad_()
    #             fwdGrad0 = recon_err_grad(fwd).detach().clone()
    #             fwd = icnn_bwd(icnn_fwd(fwd) - stepsize*fwdGrad0/5)
    #             fwd_ = fwd.cpu().detach().numpy()
    #             plt.figure()
    #             plt.imshow(fwd_[0,:,:,:].transpose(1,2,0))
    #             print("MD recon", recon_err(fwd).item())
            
    #         fwd = batch_noisy
    #         ## GRADIENT DESCENT
    #         for i in range(5):
    #             fwd.requires_grad_()
    #             fwdGrad0 = autograd.grad(recon_err(fwd), fwd)[0].detach()
    #             #fwd = iresnet_model.inverse(icnn(fwd) - stepsize*fwdGrad0) + icnn(fwd) - stepsize*fwdGrad0
    #             fwd = fwd - stepsize*fwdGrad0
    #             #fwd = icnn(fwd)
    #             print("GD recon", recon_err(fwd).item())
    #             #fwd = (fwd-fwd.min())/(fwd.max()-fwd.min())
                
    #         fwd_ = fwd.cpu().detach().numpy()
    #         #fwd_ = (fwd_-fwd_.min())/(fwd_.max()-fwd_.min())
    #         plt.figure()
    #         plt.imshow(fwd_[0,:,:,:].transpose(1,2,0))
    #         break

#%%
# save "trained" model
#torch.save(icnn_fwd.state_dict(), 'icnn_icnn_fwd_stl10_trainattempt_epoch500_strongconv0-1')
#torch.save(icnn_bwd.state_dict(), 'icnn_icnn_bwd_stl10_trainattempt_epoch500_strongconv0-1')
#%%

#icnn_fwd.load_state_dict(torch.load('icnn_icnn_fwd_stl10_trainattempt_epoch300_strongconv0-1'))
#icnn_bwd.load_state_dict(torch.load('icnn_icnn_bwd_stl10_trainattempt_epoch300_strongconv0-1'))

# Epoch 300+119: 320104
# Epoch 300+125: 315k
# Epoch 300+147: 312k
# Epoch 300+170: 305k