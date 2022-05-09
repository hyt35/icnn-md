# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:13:28 2022

@author: hongy
"""
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as f
import time
from datetime import datetime
import logging
import parse_import
import torch.autograd as autograd
device='cuda'

checkpoint_path = '/local/scratch/public/hyt35/ICNN-MD/ICNN-ARTIFICIAL/checkpoints/svm_margin/3000'
mnist_data = torchvision.datasets.MNIST('/local/scratch/public/hyt35/datasets', train=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

torch.cuda.set_device(1)
classes=(4,9)

mnist_data_filtered_idx = (mnist_data.targets == classes[0]) | (mnist_data.targets == classes[1])
mnist_data.data, mnist_data.targets = mnist_data.data[mnist_data_filtered_idx], mnist_data.targets[mnist_data_filtered_idx]

def gaussian_kernel(x,w):
    return torch.exp(x[:,None]-w[None,:])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = f.relu(f.max_pool2d(self.conv1(x), 2))
        x = f.relu(f.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = f.relu(self.fc1(x))
        x = f.dropout(x, training=self.training)
        x = self.fc2(x)
        return f.log_softmax(x)

class NetWithoutLast(nn.Module):
    def __init__(self, net):
        super(NetWithoutLast, self).__init__()
        self.conv1 = net.conv1
        self.conv2 = net.conv2
        self.conv2_drop = net.conv2_drop
        self.fc1 = net.fc1

    def forward(self, x):
        x = f.relu(f.max_pool2d(self.conv1(x), 2))
        x = f.relu(f.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = f.relu(self.fc1(x))
        return x

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

# load feature mapper
network = Net()
network.load_state_dict(torch.load('mnist_2layernet'))
pre_net = NetWithoutLast(network).to(device)
pre_net.eval()
del network

# Initialize models
dim = 51

icnn_couple = ICNNCouple(stepsize_init = 0.01, num_iters = 10, stepsize_clamp = (0.001,0.1))
icnn_couple.init_fwd(in_dim = dim, hidden = 64, num_layers=4, strong_convexity = 0.5)
icnn_couple.fwd_model.initialize_weights()
icnn_couple.init_bwd(in_dim = dim, hidden = 96, num_layers=4, strong_convexity = 1)
icnn_couple.load_state_dict(torch.load(checkpoint_path))

margin_loss = torch.nn.MultiMarginLoss()
ones = torch.tensor(1)
ones.requires_grad_(False)


#%%
dataset_size = 1000 # number of random class pair to extract
bsize = 2000 # number of random SVM init
stepsize = 0.01
icnn_couple.eval()
opt = torch.optim.Adam(icnn_couple.parameters(),lr=1e-5,betas=(0.9,0.99))

closeness_reg = 1.0
closeness_update_nepochs = 50
closeness_update_multi = 1.05

ind1 = classes[0]
ind2 = classes[1]

train_dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=dataset_size, shuffle=True)
class_lossfn = torch.nn.MarginRankingLoss(reduction='sum', margin=1)


#ind1,ind2 = torch.randperm(10)[0:2]
stepsize_scales = [1/2,1,4,10]
#stepsize_scales = [1]
for idx, (data, target) in enumerate(train_dataloader):
    data1 = data[target==ind1]
    data2 = data[target==ind2]
    len1 = len(data1)
    len2 = len(data2)

    target1 = torch.ones(data1.shape[0]) # 4 is positive
    target2 = -torch.ones(data2.shape[0]) # 9 is negative

    dat = torch.concat([data1, data2]).to(device) # (count,1,28,28), where count = dataset_size
    targ = torch.concat([target1, target2]).repeat(bsize,1).T.to(device) # (count, bsize)

    feat = pre_net(dat) # (count, 50), NN features

    # define objective functions
    def svm_loss(wb):
        w = wb[:,:-1] # (bsize, 50)
        b = wb[:,-1] # (bsize)
        skew = torch.inner(feat, w) + b # (count, bsize), probably
        return torch.sum(w**2)/2 + 2*class_lossfn(skew.flatten(), torch.zeros_like(skew.flatten()), targ.flatten())/(bsize)

    def svm_loss_grad(wb):
    #wb = torch.hstack([w, b[:,None]])
        return autograd.grad(svm_loss(wb), wb)[0]
    
    def classif_acc(wb):
        correct=0
        w = wb[:,:-1] # (bsize, 50)
        b = wb[:,-1] # (bsize)
        skew = torch.inner(feat, w) + b # (count, bsize), probably
        pred = (skew*targ)>0

        correct = pred*1.0
        correct_prob = correct.mean()
        return correct_prob

    init_wb = torch.randn(bsize, 51, requires_grad=True).to(device) # initialize bsize number of random svm initialization

    fwd = init_wb


    fig_loss, ax_loss = plt.subplots(figsize=(12,10))
    #fig_loss.suptitle("Test margin loss")
    ax_loss.set_yscale('log')
    ax_loss.set_xlabel('Iterations')
    ax_loss.set_ylabel('Cross entropy loss')

    fig_prob, ax_prob = plt.subplots(figsize = (12,10))
    #fig_prob.suptitle("Accuracy")
    ax_prob.set_xlabel('Iterations')
    ax_prob.set_ylabel('Accuracy')


    initloss = svm_loss(fwd).item()
    initacc = classif_acc(fwd).item()

    mdloss = [initloss]
    mdacc = [initacc]
    avgss = torch.mean(icnn_couple.stepsize).item()

    for ss in icnn_couple.stepsize:
        #fwd = fwd.clamp(0,1)
        fwd.requires_grad_()
        fwdGrad0 = svm_loss_grad(fwd).detach().clone()
        fwd = icnn_couple.bwd_model(icnn_couple.fwd_model(fwd) - ss*fwdGrad0).detach()
        #fwd = icnn_bwd(icnn_fwd(fwd))
        #fwd_ = fwd.cpu().detach().numpy()
        #print("MD ", svm_loss(fwd).item())
        mdloss.append(svm_loss(fwd).item())
        mdacc.append(classif_acc(fwd).item())
    for _ in range(10):
        fwd.requires_grad_()
        fwdGrad0 = svm_loss_grad(fwd).detach().clone()
        fwd = icnn_couple.bwd_model(icnn_couple.fwd_model(fwd) - avgss*fwdGrad0).detach()
        #fwd = icnn_bwd(icnn_fwd(fwd))
        #fwd_ = fwd.cpu().detach().numpy()
        #print("MD ext", svm_loss(fwd).item())
        mdloss.append(svm_loss(fwd).item())
        mdacc.append(classif_acc(fwd).item())
    ax_loss.plot(mdloss, label = "md adaptive", color='b', marker='^')
    ax_prob.plot(mdacc, label='md adaptive', color='b', marker='^')
    print(mdacc)


    max_mdloss = max(mdloss[:10])
    min_mdloss = min(mdloss)

    for stepsize_scale in stepsize_scales:
        torch.cuda.empty_cache()
        fwd = init_wb.requires_grad_()

        mdloss = [initloss]
        mdprob = [initacc]
        ## MD Fixed stepsize
        for i in range(20):
            fwd.requires_grad_()
            fwdGrad0 = svm_loss_grad(fwd).detach().clone()
            fwd = icnn_couple.bwd_model(icnn_couple.fwd_model(fwd) - stepsize*stepsize_scale*fwdGrad0).detach()
            #fwd = icnn_bwd(icnn_fwd(fwd))
            fwd_ = fwd.cpu().detach().numpy()
            #print("MD recon", svm_loss(fwd).item())
            mdloss.append(svm_loss(fwd).item())
            mdprob.append(classif_acc(fwd).item())


        gdloss = [initloss]
        gdprob = [initacc]
        ## GD
        fwd = init_wb.requires_grad_()
        for i in range(20):
            fwd.requires_grad_()
            fwdGrad0 = svm_loss_grad(fwd)
            fwd = fwd - stepsize*stepsize_scale*fwdGrad0
            #print("GD recon", svm_loss(fwd).item())
            gdloss.append(svm_loss(fwd).item())
            gdprob.append(classif_acc(fwd).item())

        ## ADAM
        adamloss = []
        adamprob = []
        tmp = init_wb
        par = tmp.clone().detach()
        par.requires_grad_()
        optimizer = torch.optim.Adam([par], lr=stepsize_scale*0.05)
        for i in range(21):
            optimizer.zero_grad()
            loss = svm_loss(par)
            adamloss.append(loss.item())
            adamprob.append(classif_acc(par).item())
            loss.backward(retain_graph=True)
            optimizer.step()
            
            #print("adam recon", svm_loss(par).item())

            #fwd = (fwd-fwd.min())/(fwd.max()-fwd.min())

        ax_loss.plot(mdloss, label = "md " + str(stepsize_scale), color = 'm', marker='x')
        ax_loss.plot(gdloss, label = "gd " + str(stepsize_scale), color = 'r', marker='o', alpha=0.7)
        ax_loss.plot(adamloss, label = "adam " + str(stepsize_scale), color = 'g', marker='v', alpha=0.7)
        ax_prob.plot(mdprob, label = "md " + str(stepsize_scale), color = 'm', marker='x')
        ax_prob.plot(gdprob, label = "gd " + str(stepsize_scale), color = 'r', marker='o', alpha=0.7)
        ax_prob.plot(adamprob, label = "adam " + str(stepsize_scale), color = 'g', marker='v', alpha=0.7)
    ax_loss.legend(loc='upper right', fontsize='x-small')
    ax_prob.legend(loc='lower right', fontsize='x-small')
    #ax_loss.set_ylim((min_mdloss/5,max_mdloss*10))
    #plt.ylim((min_mdloss/5,min_mdloss*10))
    
    break
fig_loss.savefig('figs/svm_margin_log_s')
fig_prob.savefig('figs/svm_margin_acc_mean_s')
print(icnn_couple.stepsize)
          