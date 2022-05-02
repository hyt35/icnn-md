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

checkpoint_path = '/local/scratch/public/hyt35/ICNN-MD/ICNN-ARTIFICIAL/checkpoints/nn/5000'

mnist_data = torchvision.datasets.MNIST('/local/scratch/public/hyt35/datasets', train=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
torch.cuda.set_device(0)

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
dim = 500

icnn_couple = ICNNCouple(stepsize_init = 0.01, num_iters = 10, stepsize_clamp = (0.001,0.1))
icnn_couple.init_fwd(in_dim = dim, hidden = 128, num_layers=5, strong_convexity = 0.5)
icnn_couple.fwd_model.initialize_weights()
icnn_couple.init_bwd(in_dim = dim, hidden = 196, num_layers=5, strong_convexity = 1)
icnn_couple.load_state_dict(torch.load(checkpoint_path))

#%%
dataset_size = 2000 # number of random class pair to extract
bsize = 2000 # number of random final-layer init
stepsize = 0.01
icnn_couple.train()
opt = torch.optim.Adam(icnn_couple.parameters(),lr=1e-5,betas=(0.9,0.99))

closeness_reg = 1.0
closeness_update_nepochs = 50
closeness_update_multi = 1.05


train_dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=dataset_size, shuffle=True)
class_lossfn = torch.nn.CrossEntropyLoss(reduction='sum')

stepsize_scales = [1/4,1/2,1,2,4]



for idx, (data, target) in enumerate(train_dataloader):

        dat = data.to(device) # (count,1,28,28), where count = dataset_size
        targ = target.repeat(bsize,1).to(device) # of size (bsize, count)
        feat = pre_net(dat).to(device) # (count, 50), NN features

        init_layer_mat = torch.randn(bsize, 50,10).to(device)
        # define objective functions
        def nn_loss(layer_mat): # expected of size (C, 50, 10)
        # layer mat is of size (bsize, 50,10)
        # torch.matmul(feat, final_layer) returns of form (bsize, count, 10)
        # cross entropy requires probabilities to be in dim 1 => transpose dim 1 and 2
            return class_lossfn(torch.matmul(feat, layer_mat).permute(0,2,1), targ)

        def nn_loss_grad(layer_mat):
        #wb = torch.hstack([w, b[:,None]])
            return autograd.grad(nn_loss(layer_mat), layer_mat)[0]
        def classif_acc(layer_mat):
            correct=0
            output = torch.matmul(feat, layer_mat)
            pred = output.max(2)[1] # of size (bsize, count), maybe
            correct += pred.eq(targ).sum(1) # correct, of size (bsize) so we have layerwise
            correct_prob = correct/(dataset_size)
            correct_prob_adjusted = correct_prob.mean() # postprocessing, eg max, min, mean
            return correct_prob_adjusted

        fig_loss, ax_loss = plt.subplots(figsize=(12,10))
        fig_loss.suptitle("Test cross-entropy loss")
        ax_loss.set_yscale('log')

        fig_prob, ax_prob = plt.subplots(figsize = (12,10))
        fig_prob.suptitle("Test accuracy")







        initloss = nn_loss(init_layer_mat).item()
        initacc = classif_acc(init_layer_mat).item()
        fwd = init_layer_mat.flatten(1).requires_grad_()
        mdloss = [initloss]
        mdacc = [initacc]
        avgss = torch.mean(icnn_couple.stepsize)

        for ss in icnn_couple.stepsize:
            #fwd = fwd.clamp(0,1)
            fwd.requires_grad_()
            fwdGrad0 = nn_loss_grad(fwd.reshape(bsize, 50,10)).detach().clone()
            fwd = icnn_couple.bwd_model(icnn_couple.fwd_model(fwd) - ss*fwdGrad0.flatten(1)).detach()
            #fwd = icnn_bwd(icnn_fwd(fwd))
            #fwd_ = fwd.cpu().detach().numpy()
            print("MD ", nn_loss(fwd.reshape(bsize, 50,10)).item())
            mdloss.append(nn_loss(fwd.reshape(bsize, 50,10)).item())
            mdacc.append(classif_acc(fwd.reshape(bsize, 50,10)).item())
        for _ in range(10):
            fwd.requires_grad_()
            fwdGrad0 = nn_loss_grad(fwd.reshape(bsize, 50,10)).detach().clone()
            fwd = icnn_couple.bwd_model(icnn_couple.fwd_model(fwd) - avgss*fwdGrad0.flatten(1)).detach()
            #fwd = icnn_bwd(icnn_fwd(fwd))
            #fwd_ = fwd.cpu().detach().numpy()
            print("MD ext", nn_loss(fwd.reshape(bsize, 50,10)).item())
            mdloss.append(nn_loss(fwd.reshape(bsize, 50,10)).item())
            mdacc.append(classif_acc(fwd.reshape(bsize, 50,10)).item())
        ax_loss.plot(mdloss, label = "mdloss adaptive", marker='^')
        ax_prob.plot(mdacc, label='mdlossadaptive', marker='^')


        max_mdloss = max(mdloss[:10])
        min_mdloss = min(mdloss)

        for stepsize_scale in stepsize_scales:
            torch.cuda.empty_cache()
            fwd = init_layer_mat.flatten(1).requires_grad_()

            mdloss = [initloss]
            mdprob = [initacc]
            ## MD Fixed stepsize
            for i in range(20):
                fwd.requires_grad_()
                fwdGrad0 = nn_loss_grad(fwd.reshape(bsize, 50,10)).detach().clone()
                fwd = icnn_couple.bwd_model(icnn_couple.fwd_model(fwd) - stepsize*stepsize_scale*fwdGrad0.flatten(1)).detach()
                #fwd = icnn_bwd(icnn_fwd(fwd))
                fwd_ = fwd.cpu().detach().numpy()
                print("MD recon", nn_loss(fwd.reshape(bsize, 50,10)).item())
                mdloss.append(nn_loss(fwd.reshape(bsize, 50,10)).item())
                mdprob.append(classif_acc(fwd.reshape(bsize, 50,10)).item())


            gdloss = [initloss]
            gdprob = [initacc]
            ## GD
            fwd = init_layer_mat.flatten(1).requires_grad_()
            for i in range(20):
                fwd.requires_grad_()
                fwdGrad0 = nn_loss_grad(fwd.reshape(bsize, 50,10))
                fwd = fwd - stepsize*stepsize_scale*fwdGrad0.flatten(1)
                print("GD recon", nn_loss(fwd.reshape(bsize, 50,10)).item())
                gdloss.append(nn_loss(fwd.reshape(bsize, 50,10)).item())
                gdprob.append(classif_acc(fwd.reshape(bsize, 50,10)).item())

            ## ADAM
            adamloss = []
            adamprob = []
            tmp = init_layer_mat.flatten(1)
            par = tmp.clone().detach()
            par.requires_grad_()
            optimizer = torch.optim.Adam([par], lr=stepsize_scale*0.05)
            for i in range(21):
                optimizer.zero_grad()
                loss = nn_loss(par.reshape(bsize, 50,10))
                adamloss.append(loss.item())
                adamprob.append(classif_acc(par.reshape(bsize, 50,10)).item())
                loss.backward(retain_graph=True)
                optimizer.step()
                
                print("adam recon", nn_loss(par.reshape(bsize, 50,10)).item())

                #fwd = (fwd-fwd.min())/(fwd.max()-fwd.min())

            ax_loss.plot(mdloss, label = "mdloss " + str(stepsize_scale), marker='x')
            ax_loss.plot(gdloss, label = "gdloss " + str(stepsize_scale), marker='o')
            ax_loss.plot(adamloss, label = "adamloss " + str(stepsize_scale), marker='v')
            ax_prob.plot(mdprob, label = "md " + str(stepsize_scale), marker='x')
            ax_prob.plot(gdprob, label = "gd " + str(stepsize_scale), marker='o')
            ax_prob.plot(adamprob, label = "adam " + str(stepsize_scale), marker='v')
        ax_loss.legend(loc='upper right', fontsize='x-small')
        ax_prob.legend(loc='lower right', fontsize='x-small')
        ax_loss.set_ylim((min_mdloss/5,max_mdloss*10))
        #plt.ylim((min_mdloss/5,min_mdloss*10))
        
        break
#fig_loss.savefig('nn_log')
fig_prob.savefig('nn_acc_mean')
print(icnn_couple.stepsize)
# basically all 0.001 which is bottom clamp