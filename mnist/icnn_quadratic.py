# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:01:37 2022

@author: hongy
"""
# python icnn_quadratic.py --num_epochs=5000 --num_batches=20 --checkpoint_freq=100
#from icnn import DenseICGN
#from denoising_nets_for_mcmc import ICNN
import numpy as np
import torch
from torch._C import LoggerBase
import torch.nn as nn
import torch.autograd as autograd
from torch.storage import T
import torchvision
import matplotlib.pyplot as plt
#from iunets import iUNet
import parse_import
import logging
from datetime import datetime
import torch.nn.functional as F
device = 'cuda'
checkpoint_path = 'checkpoints/quadratic/'
logging.basicConfig(filename=datetime.now().strftime('logs/%d-%m_%H:%M-quadratic.log'), filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger()
args=parse_import.parse_commandline_args()
torch.cuda.set_device(1)



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
    def __init__(self, in_dim = 10, hidden = 64, num_layers=10, strong_convexity = 0.5):
        super(ICNN, self).__init__()
        self.in_dim = in_dim # input dimension of data
        self.n_layers = num_layers # number of hidden layer
        self.hidden = hidden # number of nodes in hidden layers

        #these layers should have non-negative weights
        # self.wz = nn.ModuleList([nn.Conv2d(self.n_filters, self.n_filters, self.kernel_size, stride=1, padding=self.padding, padding_mode='circular', bias=False)\
        #                          for i in range(self.n_layers)])
        
        # #these layers can have arbitrary weights
        # self.wx_quad = nn.ModuleList([nn.Conv2d(self.n_in_channels, self.n_filters, self.kernel_size, stride=1, padding=self.padding, padding_mode='circular', bias=False)\
        #                          for i in range(self.n_layers+1)])
    
        # self.wx_lin = nn.ModuleList([nn.Conv2d(self.n_in_channels, self.n_filters, self.kernel_size, stride=1, padding=self.padding, padding_mode='circular', bias=True)\
        #                          for i in range(self.n_layers+1)])
        
        self.wz = nn.ModuleList([
                  nn.Linear(hidden,hidden,bias=False),
                  *[nn.Linear(hidden,hidden,bias=False) for _ in range(num_layers)],
                  nn.Linear(hidden,in_dim,bias=False)
          ])

        self.wx_quad = nn.ModuleList([
                  nn.Linear(in_dim,hidden,bias=False),
                  *[nn.Linear(in_dim,hidden,bias=False) for _ in range(num_layers+1)],
                  nn.Linear(in_dim,in_dim,bias=False)
          ])

        self.wx_lin = nn.ModuleList([
                  nn.Linear(in_dim,hidden,bias=True),
                  *[nn.Linear(in_dim,hidden,bias=True) for _ in range(num_layers+1)],
                  nn.Linear(in_dim,in_dim,bias=True)
          ])

        #slope of leaky-relu
        self.negative_slope = 0.2 
        self.strong_convexity = strong_convexity
        
        
    def scalar(self, x):
        z = torch.nn.functional.leaky_relu(self.wx_quad[0](x)**2 + self.wx_lin[0](x), negative_slope=self.negative_slope)
        for layer in range(self.n_layers+1):
            z = torch.nn.functional.leaky_relu(self.wz[layer](z) + self.wx_quad[layer+1](x)**2 + self.wx_lin[layer+1](x), negative_slope=self.negative_slope)

        return z
    
    def forward(self, x):
        foo = compute_gradient(self.scalar, x)
        return (1-self.strong_convexity)*foo + self.strong_convexity*x
    
    #a weight initialization routine for the ICNN
    def initialize_weights(self, min_val=0.0, max_val=0.001, device=device):
        for layer in range(self.n_layers):
            self.wz[layer+1].weight.data = min_val + (max_val - min_val)\
            * torch.rand(self.hidden, self.hidden).to(device)
        self.wz[0].weight.data = min_val + (max_val - min_val)\
            * torch.rand(self.hidden, self.hidden).to(device)
        self.wz[-1].weight.data = min_val + (max_val - min_val)\
            * torch.rand(self.hidden, self.in_dim).to(device)
            
        return self
    
    #a zero clipping functionality for the ICNN (set negative weights to 0)
    def zero_clip_weights(self): 
        for layer in range(self.n_layers):
            self.wz[layer].weight.data.clamp_(0)

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
        
    def init_fwd(self,in_dim = 10, hidden = 64, num_layers=10, strong_convexity = 0.5):
        self.fwd_model = ICNN(in_dim, hidden, num_layers, strong_convexity).to(device)
        return self
        
    def init_bwd(self,in_dim = 10, hidden = 64, num_layers=10, strong_convexity = 0.5):
        self.bwd_model = ICNN(in_dim, hidden, num_layers, strong_convexity).to(device)
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
dim = 10

icnn_couple = ICNNCouple(stepsize_init = 0.01, num_iters = 10, stepsize_clamp = (0.001,0.1))
icnn_couple.init_fwd(in_dim = dim, hidden = 64, num_layers=4, strong_convexity = 0.5)
icnn_couple.fwd_model.initialize_weights()
icnn_couple.init_bwd(in_dim = dim, hidden = 96, num_layers=4, strong_convexity = 1)

#%%
logger.info("fwd params"+str( sum(p.numel() for p in icnn_couple.fwd_model.parameters())))
logger.info("bwd params"+str( sum(p.numel() for p in icnn_couple.bwd_model.parameters())))
#%%
if args.from_checkpoint is not None:
    icnn_couple.load_state_dict(torch.load(args.from_checkpoint))

n_epochs = args.num_epochs
noise_level = 0.05
reg_param = 0.3
stepsize = 0.01
bsize=500
closeness_reg = 1.0
closeness_update_nepochs = 200
closeness_update_multi = 1.05
#%%
loss_fn = torch.nn.MSELoss()
gen = torch.Generator()
gen.manual_seed(0) # deterministic W

if __name__ == '__main__': 
    if args.train:
        icnn_couple.train()
        opt = torch.optim.Adam(icnn_couple.parameters(),lr=1e-5,betas=(0.9,0.99))
        total_loss = 0
        total_closeness = 0
        W = torch.randn(dim, dim, generator=gen).to(device)
        logger.info(W)
        for epoch in range(n_epochs):

            x = torch.randn(bsize, dim)
            
            b = torch.randn(dim).to(device)

            # define objective functions
            # min ||Wx-b||_2^2
            def recon_err(x):
                return torch.sum((torch.matmul(W,x.T).T - b)**2)
            
            def recon_err_grad(x):
                return autograd.grad(recon_err(x), x)[0]


            fwd = x.to(device)
            fwd.requires_grad_()
            loss = 0
            #closeness = icnn_couple.fwdbwdloss(torch.zeros_like(b).to(device))
            closeness = 0
            for stepsize in icnn_couple.stepsize:
                fwdGrad = recon_err_grad(fwd)
                fwd = icnn_couple.bwd_model(icnn_couple.fwd_model(fwd) - stepsize*fwdGrad) 
                loss += recon_err(fwd)
                closeness += icnn_couple.fwdbwdloss(fwd)

            #loss = recon_err(fwd)

            err = loss+(closeness)*closeness_reg
            
            opt.zero_grad()
            err.backward()
            opt.step()
            
            icnn_couple.clip_fwd()
            #icnn_bwd.zero_clip_weights()
            icnn_couple.clamp_stepsizes()
            
            total_loss += err.item()
            total_closeness += closeness.item()

            if(epoch % args.num_batches == args.num_batches-1):
                avg_loss = total_loss/args.num_batches/bsize
                avg_fwdbwd = total_closeness/args.num_batches/bsize
                print("curr loss", loss.item(), "curr closeness", closeness.item())
                train_log = "epoch:[{}/{}] , avg_loss = {:.4f}, avg_fwdbwd = {:.4f}".\
                    format(epoch+1, args.num_epochs, avg_loss, avg_fwdbwd)
                print(train_log)
                logger.info(train_log)
                total_loss = 0
                total_closeness = 0
            
            #print("Epoch", epoch, "total loss", total_loss, "fwdbwd", total_fwdbwd)
            # Checkpoint
            # Increase closeness regularization
            if (epoch%closeness_update_nepochs == closeness_update_nepochs-1):
                closeness_reg = closeness_reg*closeness_update_multi

            if (epoch%args.checkpoint_freq == args.checkpoint_freq-1):
                torch.save(icnn_couple.state_dict(), checkpoint_path+str(epoch+1))
            # Log
                logger.info("\n====epoch:[{}/{}], epoch_loss = {:.2f}, epoch_fwdbwd = {:.4f}====\n".format(epoch+1, args.num_epochs, total_loss, total_closeness))
