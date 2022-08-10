import torch
from torch.nn import KLDivLoss
import torch.autograd as autograd
import matplotlib.pyplot as plt
import itertools
import numpy as np
from datetime import datetime
import logging
import parse_import
import torch.autograd as autograd
device='cuda'
from pathlib import Path
import re
import itertools
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib import cm


checkpoint_path = 'checkpoints/simplex_KL/'
log_path = 'logs/%d-%m_%H%M-simplex_KL.log'
torch.cuda.set_device(2)

plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)  

args=parse_import.parse_commandline_args()
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)


lossfn = KLDivLoss(reduction='sum', log_target=False)
dim = 3

device = "cuda"


def simplexProjection(inp):
    # expect inp in shape (, 3)
    inp_sort, _ = inp.sort(descending=True)
    tmp = torch.zeros(inp.shape[0]).to(device)
    flag = torch.ones(inp.shape[0]).to(device)
    tmax=0

    for i in range(2):
        tmp = tmp + inp_sort[:,i]
        tmax = (1-flag)*tmax + flag*(tmp - 1)/(i+1)
        
        flag = (~(tmax >= inp_sort[:,i+1]))*flag

    tmax = (1-flag)*tmax+flag*(tmp+inp_sort[:,-1]-1)/3

    out = (inp-tmax[:,None]).clamp(0)
    return out # clamp to positive

def negEntropy(x):
    # must be on simplex
    # no safety check
    # can take (..., n)
    return (x*torch.log(x)).sum(-1)

def negEntropyGrad(x):
    return 1+torch.log(x)

def conjEntropy(x):
    return torch.log(torch.exp(x).sum(-1))

def conjEntropyGrad(x):
    return torch.exp(x)/(torch.exp(x).sum(-1)[:,None])

# initialize on simplex
#%%
# KL
bsize = 200

expSampler = torch.distributions.exponential.Exponential(torch.ones(bsize, 3))

# uniform sample on 2-simplex
_x = expSampler.sample()+0.01 # make sure its not degen
_x = _x/_x.sum(1)[:,None] # so when projected, get uniform sample on simplex.
x = _x.to(device) 

_b = expSampler.sample()+0.01 # make sure its not degen
_b = _b/_b.sum(1)[:,None] # so when projected, get uniform sample on simplex.
b = _b.to(device) 

def dist(x):
    #return lossfn(torch.log(x), b)
    return lossfn(torch.log(b), x)
def distGrad(x):
    return autograd.grad(dist(x), x)[0]

stepsize_scales = [1/4,1/2,1,2,4]
stepsize = 0.1

initloss = dist(x).cpu()

fig_loss, ax_loss = plt.subplots(figsize = (10.5,7))
plt.subplots_adjust(left=0.07, bottom=0.08, right=0.95, top=0.95)
#fig_loss.suptitle("Test margin loss")
ax_loss.set_yscale('log')
ax_loss.set_xlabel('Iterations')
ax_loss.set_ylabel('KL divergence')

for stepsize_scale, ms in zip(stepsize_scales, itertools.cycle('s^+*x')):
    # MD
    fwd = x
    #fwd.requires_grad_()
    mdloss = [initloss]
    for i in range(50):
        fwd.requires_grad_()
        fwdGrad0 = distGrad(fwd).detach().clone()
        fwd = conjEntropyGrad(negEntropyGrad(fwd) - stepsize*stepsize_scale*fwdGrad0/np.sqrt(i+1)).detach()
        #fwd = conjEntropyGrad(negEntropyGrad(fwd) - stepsize*stepsize_scale*fwdGrad0).detach()
        #fwd = icnn_bwd(icnn_fwd(fwd))
        fwd_ = fwd.cpu().detach().numpy()
        #print("MD recon", obj(fwd).item())
        mdloss.append(dist(fwd).cpu().item()) # this should be on simplex anyway

    gdloss = [initloss]
    # Proj sub-GD
    fwd = x
    fwd.requires_grad_()

    for i in range(50):
        fwd.requires_grad_()
        fwdGrad0 = distGrad(fwd).detach().clone()
        fwd = simplexProjection((fwd - stepsize*stepsize_scale*fwdGrad0)).detach()
        gdloss.append(dist(fwd).cpu().item()) 

    ax_loss.plot(mdloss, label = "md " + str(stepsize_scale), color = 'm', marker=ms)
    ax_loss.plot(gdloss, label = "gd " + str(stepsize_scale), color = 'r', marker=ms)

plt.suptitle("vary stepsize")
fig_loss.savefig('figs/entropy_test_KL')


#%%
# lsq
_x = expSampler.sample()+0.01 # make sure its not degen
_x = _x/_x.sum(1)[:,None] # so when projected, get uniform sample on simplex.
x = _x.to(device) 

_b=torch.rand((bsize,3))
#_b = expSampler.sample()+0.01 # make sure its not degen
#_b = _b/_b.sum(1)[:,None] # so when projected, get uniform sample on simplex.
b = _b.to(device) 

fig_loss, ax_loss = plt.subplots(figsize = (10.5,7))
plt.subplots_adjust(left=0.07, bottom=0.08, right=0.95, top=0.95)
#fig_loss.suptitle("Test margin loss")
ax_loss.set_yscale('log')
ax_loss.set_xlabel('Iterations')
ax_loss.set_ylabel('Least squares loss')
def dist(x):
    #return lossfn(torch.log(x), b)
    return torch.sum((x-b)**2)/2
def distGrad(x):
    return x-b

for stepsize_scale, ms in zip(stepsize_scales, itertools.cycle('s^+*x')):
    # MD
    fwd = x
    #fwd.requires_grad_()
    mdloss = [initloss]
    for i in range(50):
        fwd.requires_grad_()
        fwdGrad0 = distGrad(fwd).detach().clone()
        fwd = conjEntropyGrad(negEntropyGrad(fwd) - stepsize*stepsize_scale*fwdGrad0/np.sqrt(i+1)).detach()
        #fwd = conjEntropyGrad(negEntropyGrad(fwd) - stepsize*stepsize_scale*fwdGrad0).detach()
        #fwd = icnn_bwd(icnn_fwd(fwd))
        fwd_ = fwd.cpu().detach().numpy()
        #print("MD recon", obj(fwd).item())
        mdloss.append(dist(fwd).cpu().item()) # this should be on simplex anyway

    gdloss = [initloss]
    # Proj sub-GD
    fwd = x
    fwd.requires_grad_()

    for i in range(50):
        fwd.requires_grad_()
        fwdGrad0 = distGrad(fwd).detach().clone()
        fwd = simplexProjection((fwd - stepsize*stepsize_scale*fwdGrad0)).detach()
        gdloss.append(dist(fwd).cpu().item()) 

    ax_loss.plot(mdloss, label = "md " + str(stepsize_scale), color = 'm', marker=ms)
    ax_loss.plot(gdloss, label = "gd " + str(stepsize_scale), color = 'r', marker=ms)

#plt.suptitle("vary stepsize")
fig_loss.savefig('figs/entropy_test_lsq_vary')

#%%
# lsq
fig_loss, ax_loss = plt.subplots(figsize = (10.5,7))
plt.subplots_adjust(left=0.07, bottom=0.08, right=0.95, top=0.95)
#fig_loss.suptitle("Test margin loss")
ax_loss.set_yscale('log')
ax_loss.set_xlabel('Iterations')
ax_loss.set_ylabel('Least squares loss')
def dist(x):
    #return lossfn(torch.log(x), b)
    return torch.sum((x-b)**2)
def distGrad(x):
    return autograd.grad(dist(x), x)[0]

for stepsize_scale, ms in zip(stepsize_scales, itertools.cycle('s^+*x')):
    # MD
    fwd = x
    #fwd.requires_grad_()
    mdloss = [initloss]
    for i in range(50):
        fwd.requires_grad_()
        fwdGrad0 = distGrad(fwd).detach().clone()
        fwd = conjEntropyGrad(negEntropyGrad(fwd) - stepsize*stepsize_scale*fwdGrad0).detach()
        #fwd = conjEntropyGrad(negEntropyGrad(fwd) - stepsize*stepsize_scale*fwdGrad0).detach()
        #fwd = icnn_bwd(icnn_fwd(fwd))
        fwd_ = fwd.cpu().detach().numpy()
        #print("MD recon", obj(fwd).item())
        mdloss.append(dist(fwd).cpu().item()) # this should be on simplex anyway

    gdloss = [initloss]
    # Proj sub-GD
    fwd = x
    fwd.requires_grad_()

    for i in range(50):
        fwd.requires_grad_()
        fwdGrad0 = distGrad(fwd).detach().clone()
        fwd = simplexProjection((fwd - stepsize*stepsize_scale*fwdGrad0)).detach()
        gdloss.append(dist(fwd).cpu().item()) 

    ax_loss.plot(mdloss, label = "md " + str(stepsize_scale), color = 'm', marker=ms)
    ax_loss.plot(gdloss, label = "gd " + str(stepsize_scale), color = 'r', marker=ms)

plt.suptitle("fix stepsize lsq")
fig_loss.savefig('figs/entropy_test_lsq_fix')