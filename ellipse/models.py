import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torchvision
import matplotlib.pyplot as plt
#from iunets import iUNet
import parse_import
import logging
from datetime import datetime
import torch.nn.functional as F



def compute_gradient(net, inp, dev):
    inp_with_grad = inp.requires_grad_(True)
    out = net(inp_with_grad)  
    fake = torch.tensor(np.ones(out.shape), device=dev).requires_grad_(False)
    # Get gradient w.r.t. input
    gradients = autograd.grad(outputs=out, inputs=inp_with_grad,
                              grad_outputs=fake, create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    return gradients

class ICNN(nn.Module):
    def __init__(self, num_in_channels=1, num_filters=64, kernel_dim=5, num_layers=10, strong_convexity = 0.5, imsize=32*32, dense_size = 200, device='cuda'):
        super(ICNN, self).__init__()
        self.n_in_channels = num_in_channels
        self.n_layers = num_layers
        self.n_filters = num_filters
        self.kernel_size = kernel_dim
        self.imsize = imsize
        self.padding = (self.kernel_size-1)//2
        self.dense_size = dense_size
        self.conv_filters=5

        #these layers should have non-negative weights
        self.wz = nn.ModuleList([nn.Conv2d(self.n_filters, self.n_filters, self.kernel_size, stride=1, padding=self.padding, padding_mode='circular', bias=False)\
                                 for i in range(self.n_layers)])
        
        #these layers can have arbitrary weights
        self.wx_quad = nn.ModuleList([nn.Conv2d(self.n_in_channels, self.n_filters, self.kernel_size, stride=1, padding=self.padding, padding_mode='circular', bias=False)\
                                 for i in range(self.n_layers+1)])
    
        self.wx_lin = nn.ModuleList([nn.Conv2d(self.n_in_channels, self.n_filters, self.kernel_size, stride=1, padding=self.padding, padding_mode='circular', bias=True)\
                                 for i in range(self.n_layers+1)])
        
        #one final conv layer with nonnegative weights
        self.final_conv2d = nn.Conv2d(self.n_filters, self.conv_filters, self.kernel_size, stride=2, padding=self.padding, padding_mode='circular', bias=False)
        
        # img size is 32x32
        self.dense1 = nn.Linear(self.conv_filters*imsize//(2**self.n_layers), self.conv_filters*self.dense_size//2, bias=False)
        self.dense2 = nn.Linear(self.conv_filters*self.dense_size//2, self.conv_filters*self.dense_size//4, bias=False)
        #slope of leaky-relu
        self.negative_slope = 0.2 
        self.strong_convexity = strong_convexity
        self.device = device
        
    def scalar(self, x):
        z = torch.nn.functional.leaky_relu(self.wx_quad[0](x)**2 + self.wx_lin[0](x), negative_slope=self.negative_slope)
        for layer in range(self.n_layers):
            z = torch.nn.functional.leaky_relu(self.wz[layer](z) + self.wx_quad[layer+1](x)**2 + self.wx_lin[layer+1](x), negative_slope=self.negative_slope)
        z = self.final_conv2d(z)
        #print(z.shape)
        z = z.view(z.size()[0],-1)
        z = torch.nn.functional.leaky_relu(self.dense1(z), negative_slope = self.negative_slope)
        z = self.dense2(z)
        z_avg = torch.nn.functional.avg_pool1d(z, z.size(-1))
        #z_avg = torch.nn.functional.avg_pool2d(z, z.size()[2:]).view(z.size()[0], -1)
        
        return z_avg + .5 * self.strong_convexity * (x ** 2).sum(dim=[1,2,3]).reshape(-1, 1)
    
    def forward(self, x):
        foo = compute_gradient(self.scalar, x, dev=self.device)
        return foo
    
    
    #a zero clipping functionality for the ICNN (set negative weights to 0)
    def zero_clip_weights(self): 
        for layer in range(self.n_layers):
            self.wz[layer].weight.data.clamp_(0)
        
        self.final_conv2d.weight.data.clamp_(0)
        
        self.dense1.weight.data.clamp_(0)
        self.dense2.weight.data.clamp_(0)
        return self


class ICNNCouple(nn.Module):
    def __init__(self, device, imsize = 128*128, stepsize_init = 0.01, num_iters = 10, stepsize_clamp = (0.001,0.1)):
        super(ICNNCouple, self).__init__()
        self.fwd_model = None
        self.bwd_model = None
        self.stepsize = nn.Parameter(stepsize_init * torch.ones(num_iters).to(device))
        self.num_iters = num_iters
        self.ssmin = stepsize_clamp[0]
        self.ssmax = stepsize_clamp[1]
        self.device = device
        self.imsize = imsize
        # Logger
        # Checkpoint
        
    def init_fwd(self, num_in_channels=1, num_filters=64, kernel_dim=5, num_layers=10, strong_convexity = 0.5, dense_size = 200):
        self.fwd_model = ICNN(num_in_channels, num_filters, kernel_dim, num_layers, strong_convexity, imsize = self.imsize, dense_size = dense_size, device=self.device).to(self.device)
        return self
        
    def init_bwd(self, num_in_channels=1, num_filters=64, kernel_dim=5, num_layers=10, strong_convexity = 0.5, dense_size = 200):
        self.bwd_model = ICNN(num_in_channels, num_filters, kernel_dim, num_layers, strong_convexity, imsize = self.imsize, dense_size = dense_size, device=self.device).to(self.device)
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
        if self.training:
            iterates = []
            fwd = x
            for ss in self.stepsize:
                fwd = self.bwd_model(self.fwd_model(fwd) - ss*gradFun(fwd)) 
                iterates.append(fwd)
        else:
            iterates = []
            fwd = x
            for ss in self.stepsize:
                fwd = fwd.detach()
                fwd = self.bwd_model(self.fwd_model(fwd) - ss*gradFun(fwd)) 
                iterates.append(fwd.clone().detach())

        return iterates
    
    def clamp_stepsizes(self):
        with torch.no_grad():
            self.stepsize.clamp_(self.ssmin,self.ssmax)
        return self
            
    def fwdbwdloss(self, x):
        return torch.linalg.vector_norm(self.bwd_model(self.fwd_model(x))-x, ord=1)

class ICNNCoupleMomentum(nn.Module):
    def __init__(self, device, imsize = 128*128, stepsize_init = 0.01, num_iters = 10, stepsize_clamp = (0.001,0.1), r=3, gamma=1):
        super(ICNNCoupleMomentum, self).__init__()
        self.fwd_model = None
        self.bwd_model = None
        self.stepsize = nn.Parameter(stepsize_init * torch.ones(num_iters).to(device))
        self.num_iters = num_iters
        self.ssmin = stepsize_clamp[0]
        self.ssmax = stepsize_clamp[1]
        self.device = device
        self.imsize = imsize
        self.r = r
        self.gamma = gamma
        # Logger
        # Checkpoint
        
    def init_fwd(self, num_in_channels=1, num_filters=64, kernel_dim=5, num_layers=10, strong_convexity = 0.5, dense_size = 200):
        self.fwd_model = ICNN(num_in_channels, num_filters, kernel_dim, num_layers, strong_convexity, imsize = self.imsize, device=self.device, dense_size = dense_size).to(self.device)
        return self
        
    def init_bwd(self, num_in_channels=1, num_filters=64, kernel_dim=5, num_layers=10, strong_convexity = 0.5, dense_size = 200):
        self.bwd_model = ICNN(num_in_channels, num_filters, kernel_dim, num_layers, strong_convexity, imsize = self.imsize, device=self.device, dense_size = dense_size).to(self.device)
        return self
    
    def clip_fwd(self):
        self.fwd_model.zero_clip_weights()
        return self
        
    def clip_bwd(self):
        self.bwd_model.zero_clip_weights()
        return self
        
    # https://papers.nips.cc/paper/2015/hash/f60bb6bb4c96d4df93c51bd69dcc15a0-Abstract.html
    # accelerate
    def forward(self, x, gradFun = None):
        if gradFun is None:
            raise RuntimeError("Gradient function not provided")
        if self.training:
            iterates = []
            xktilde = x.clone().detach()
            zktilde = x.clone().detach()
            xk = x.clone().detach()
            currentstep = 1
            for ss in self.stepsize:
                zktilde = self.bwd_model(self.fwd_model(xk) - currentstep*ss*gradFun(xk)/self.r) 
                xktilde = xk - self.gamma*ss*gradFun(xk) # gradient term with R = euclidean
                lambdak = self.r/(self.r + currentstep)


                xk = lambdak * zktilde + (1-lambdak) * xktilde
                iterates.append(xk)
                currentstep += 1

        else:
            iterates = []
            xktilde = x.clone().detach()
            zktilde = x.clone().detach()
            xk = x.clone().detach()
            currentstep = 1
            for ss in self.stepsize:
                xk = xk.detach()
                zktilde = self.bwd_model(self.fwd_model(xk) - currentstep*ss*gradFun(xk)/self.r) 
                xktilde = xk - self.gamma*ss*gradFun(xk) # gradient term with R = euclidean
                lambdak = self.r/(self.r + currentstep)


                xk = lambdak * zktilde + (1-lambdak) * xktilde
                iterates.append(xk.clone().detach())
                currentstep += 1
        return iterates
    
    def clamp_stepsizes(self):
        with torch.no_grad():
            self.stepsize.clamp_(self.ssmin,self.ssmax)
        return self
            
    def fwdbwdloss(self, x):
        return torch.linalg.vector_norm(self.bwd_model(self.fwd_model(x))-x, ord=1)