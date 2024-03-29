
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
import itertools
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib import cm
from torch.nn import KLDivLoss


PLOTLOSS=True
PLOTFWD=True
device='cuda'

checkpoint_path = '/local/scratch/public/hyt35/ICNN-MD/ICNN-ARTIFICIAL/checkpoints/simplex_KL2/10000'

torch.cuda.set_device(0)
args=parse_import.parse_commandline_args()
lossfn = KLDivLoss(reduction='sum', log_target=False)
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)  
def gaussian_kernel(x,w):
    return torch.exp(x[:,None]-w[None,:])


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
                  *[nn.Linear(hidden,hidden,bias=False) for _ in range(num_layers)],
                  nn.Linear(hidden,out_features = 1,bias=False)
          ])
        self.wx_quad = nn.ModuleList([
                  *[nn.Linear(in_dim,hidden,bias=False) for _ in range(num_layers+1)],
                  nn.Linear(in_dim,1,bias=False)
          ])

        self.wx_lin = nn.ModuleList([
                  *[nn.Linear(in_dim,hidden,bias=True) for _ in range(num_layers+1)],
                  nn.Linear(in_dim,1,bias=False)
          ])
        

        #slope of leaky-relu
        self.negative_slope = 0.2 
        self.strong_convexity = strong_convexity
        
        
    def scalar(self, x):
        z = torch.nn.functional.leaky_relu(self.wx_quad[0](x)**2 + self.wx_lin[0](x), negative_slope=self.negative_slope) # initial z
        for layer in range(self.n_layers-1):
            z = torch.nn.functional.leaky_relu(self.wz[layer](z) + self.wx_quad[layer+1](x)**2 + self.wx_lin[layer+1](x), negative_slope=self.negative_slope)
        z = self.wz[-1](z) + self.wx_quad[-1](x)**2 + self.wx_lin[-1](x)
        return z
    
    def forward(self, x):
        foo = compute_gradient(self.scalar, x)
        return (1-self.strong_convexity)*foo + self.strong_convexity*x
    
    #a weight initialization routine for the ICNN
    def initialize_weights(self, mean=-4.0, std=0.1, device=device):
        for core in self.wz:
            core.weight.data.normal_(mean,std).exp_()
            
        return self
    
    #a zero clipping functionality for the ICNN (set negative weights to 0)
    def zero_clip_weights(self): 
        for core in self.wz:
            core.weight.data.clamp_(0)

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
    return out
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

dim=3

icnn_couple = ICNNCouple(stepsize_init = 0.5, num_iters = 10, stepsize_clamp = (0.1,1.0))
icnn_couple.init_fwd(in_dim = dim, hidden = 30, num_layers=4, strong_convexity = 0.2)
icnn_couple.fwd_model.initialize_weights()
icnn_couple.init_bwd(in_dim = dim, hidden = 40, num_layers=4, strong_convexity = 0.2)
icnn_couple.load_state_dict(torch.load(checkpoint_path))
icnn_couple.eval()

n_epochs = args.num_epochs
noise_level = 0.05
reg_param = 0.3
stepsize = 0.1
bsize=2000
closeness_reg = 1.0
closeness_update_nepochs = 400
closeness_update_multi = 1.05
#%%
expSampler = torch.distributions.exponential.Exponential(torch.ones(bsize, 3))

if PLOTLOSS:
    loss_fn = torch.nn.MSELoss()
    gen = torch.Generator()
    gen.manual_seed(0) # deterministic W


    
    opt = torch.optim.Adam(icnn_couple.parameters(),lr=1e-5,betas=(0.9,0.99))
    total_loss = 0
    total_closeness = 0
    #W = torch.Tensor([[[2,1],[1,2]]]).to(device)

    stepsize_scales = [1/4,1/2,1,2,4]
    #stepsize_scales = [1]


    _x = expSampler.sample()
    _x = _x/_x.sum(1)[:,None] # so when projected, get uniform sample on simplex.
    x = _x.to(device)
    x = simplexProjection(x) # only used for testing
  
    _b = expSampler.sample()
    _b = _b/_b.sum(1)[:,None] # so when projected, get uniform sample on simplex.
    b = _b.to(device)
    b_simplex = simplexProjection(b) # only used for testing
    # define objective functions
    # min ||Wx-b||_2^2
    def obj(x):
        #return lossfn(torch.log(x), b)
        return lossfn(torch.log(b), x.clamp(0.001))
    def obj_grad(x):
        return autograd.grad(obj(x), x)[0]


    fwd = x.to(device)


    fig_loss, ax_loss = plt.subplots(figsize = (10.5,7))
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.9, top=0.9)
    #fig_loss.suptitle("Test margin loss")
    ax_loss.set_yscale('log')
    ax_loss.set_xlabel('Iterations')
    ax_loss.set_ylabel('KL divergence')



    initloss = obj(fwd).item()

    mdloss = [initloss]

    avgss = torch.min(icnn_couple.stepsize).item()

    for ss in icnn_couple.stepsize:
        #fwd = fwd.clamp(0,1)
        fwd.requires_grad_()
        fwdGrad0 = obj_grad(fwd).detach().clone()
        fwd = icnn_couple.bwd_model(icnn_couple.fwd_model(fwd) - ss*fwdGrad0)
        #fwd = icnn_couple.bwd_model(icnn_couple.fwd_model(fwd) - ss*fwdGrad0)
        #fwd = icnn_bwd(icnn_fwd(fwd))
        #fwd_ = fwd.cpu().detach().numpy()
        #print("MD ", svm_loss(fwd).item())
        mdloss.append(obj(simplexProjection(fwd)).item())
        #print(obj(fwd))

    
    foofig = plt.figure()
    fooax = foofig.add_subplot(projection='3d')
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
    # bar = simplexProjection(fwd).detach().cpu().numpy()
    bar = fwd.detach().cpu().numpy()
    fooax.scatter(bar[:,0], bar[:,1], bar[:,2])
    foobar = b_simplex.detach().cpu().numpy()
    fooax.scatter(foobar[:,0], foobar[:,1],foobar[:,2], alpha=0.05)
    
    foofig.savefig('figs/KL2_iters')

    for _ in range(40):
        fwd.requires_grad_()
        fwdGrad0 = obj_grad(fwd).detach().clone()
        fwd = icnn_couple.bwd_model(icnn_couple.fwd_model(fwd) - avgss*fwdGrad0).detach()
        #fwd = simplexProjection(icnn_couple.bwd_model(icnn_couple.fwd_model(fwd) - avgss*fwdGrad0).detach())
        #fwd = icnn_bwd(icnn_fwd(fwd))
        #fwd_ = fwd.cpu().detach().numpy()
        #print("MD ext", obj(fwd).item())
        mdloss.append(obj(simplexProjection(fwd)).item())

    ax_loss.plot(mdloss, label = "md adaptive", color='b', marker='^')



    max_mdloss = max(mdloss[:10])
    min_mdloss = min(mdloss)

    for stepsize_scale, ms in zip(stepsize_scales, itertools.cycle('>^+*x')):
        torch.cuda.empty_cache()
        fwd = x.to(device).requires_grad_()

        mdloss = [initloss]

        ## MD Fixed stepsize
        for i in range(50):
            fwd.requires_grad_()
            fwdGrad0 = obj_grad(fwd).detach().clone()
            fwd = icnn_couple.bwd_model(icnn_couple.fwd_model(fwd) - stepsize*stepsize_scale*fwdGrad0).detach()
            #fwd = icnn_bwd(icnn_fwd(fwd))
            fwd_ = fwd.cpu().detach().numpy()
            #print("MD recon", obj(fwd).item())
            mdloss.append(obj(simplexProjection(fwd)).item())



        gdloss = [initloss]

        ## GD
        fwd =  x.to(device).requires_grad_()
        for i in range(50):
            fwd.requires_grad_()
            fwdGrad0 = obj_grad(fwd)
            fwd = simplexProjection(fwd - stepsize*stepsize_scale*fwdGrad0)
            #print("GD recon", obj(fwd).item())
            gdloss.append(obj(fwd).item())
            #print(fwd[0,:])
        ## ADAM
        # adamloss = []
        # tmp =  x.to(device)
        # par = tmp.clone().detach()
        # par.requires_grad_()
        # optimizer = torch.optim.Adam([par], lr=stepsize_scale*0.05)
        # for i in range(21):
        #     optimizer.zero_grad()
        #     loss = obj(par)
        #     adamloss.append(loss.item())

        #     loss.backward(retain_graph=True)
        #     optimizer.step()
            
        #     #print("adam recon", obj(par).item())

        #     #fwd = (fwd-fwd.min())/(fwd.max()-fwd.min())
        fwd = x
        #fwd.requires_grad_()
        entropymdloss = [initloss]
        for i in range(50):
            fwd.requires_grad_()
            fwdGrad0 = obj_grad(fwd).detach().clone()
            fwd = conjEntropyGrad(negEntropyGrad(fwd) - stepsize*stepsize_scale*fwdGrad0).detach()
            #fwd = conjEntropyGrad(negEntropyGrad(fwd) - stepsize*stepsize_scale*fwdGrad0).detach()
            #fwd = icnn_bwd(icnn_fwd(fwd))
            fwd_ = fwd.cpu().detach().numpy()
            #print("MD recon", obj(fwd).item())
            entropymdloss.append(obj(fwd).cpu().item()) # this should be on simplex anyway

            ax_loss.plot(mdloss, label = "md " + str(stepsize_scale), color = 'm', marker=ms)
            ax_loss.plot(gdloss, label = "gd " + str(stepsize_scale), color = 'r', marker=ms)
            ax_loss.plot(entropymdloss, label = "entropy md " + str(stepsize_scale), color = 'g', marker=ms)
        #ax_loss.plot(adamloss, label = "adam " + str(stepsize_scale), color = 'g', marker=ms)

    #ax_loss.set_ylim((min_mdloss/5,max_mdloss*10))
    #plt.ylim((min_mdloss/5,min_mdloss*10))


    #ax_loss.set_xticks(np.arange(0,21,step=5))


    md_adap_patch = mpatches.Patch(color='b', label='Adaptive LMD')
    md_patch = mpatches.Patch(color='m', label='LMD')
    gd_adap_patch = mpatches.Patch(color='r', label='GD')
    entropy_adap_patch = mpatches.Patch(color='g', label='Entropy MD')

    box = ax_loss.get_position()
    ax_loss.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    legend1 = ax_loss.legend(handles=[md_adap_patch,md_patch,gd_adap_patch,entropy_adap_patch], bbox_to_anchor=(1.02, 0.7),
                            loc='center left', borderaxespad=0.)
    #[1/2,1,2,4,10]
    #'>^+*x'

    _mrkempty =  mlines.Line2D([], [], color='black', marker='', linestyle='None',label='Step-size multi')
    _mrk1 = mlines.Line2D([], [], color='black', marker='>', linestyle='None',
                            markersize=10, label='1/4')
    _mrk2 = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                            markersize=10, label='1/2')
    _mrk3 = mlines.Line2D([], [], color='black', marker='+', linestyle='None',
                            markersize=10, label='1')
    _mrk4 = mlines.Line2D([], [], color='black', marker='*', linestyle='None',
                              markersize=10, label='2')
    _mrk5 = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                              markersize=10, label='4')

    ax_loss.legend(handles=[_mrkempty, _mrk1,_mrk2,_mrk3,_mrk4,_mrk5], bbox_to_anchor=(1.02,0.3),
                            loc='center left', borderaxespad=0.)
    ax_loss.add_artist(legend1)

    #ax_loss.legend(loc='upper right', fontsize='x-small')
    #ax_prob.legend(loc='lower right', fontsize='x-small')
    fig_loss.savefig('figs/KL2_loss')
    fig_loss.savefig('figs/KL2_loss.svg')
    fig_loss.savefig('figs/KL2_loss.pdf')


if PLOTFWD:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
    X = np.arange(0, 1, 0.02)
    Y = np.arange(0, 1, 0.02)
    X, Y = np.meshgrid(X, Y)

    
    #foo = X**2+Y**2 +1e-3
    #foo = (2*X+Y)**2+(X+2*Y)**2+1e-1
    _X = torch.Tensor(X)
    _Y = torch.Tensor(Y)
    allow = (_X+_Y)<=1
    _X = _X[allow]
    _Y = _Y[allow]

    print(_X.shape)
    print(_Y.shape)
    
    Z = icnn_couple.fwd_model.scalar(torch.dstack((_X,_Y, 1-_X-_Y)).to(device)).detach().cpu().numpy()[0,:,0]
    #Z = icnn_couple.fwd_model.scalar(torch.dstack((_X,_Y, 1-_X-_Y)).to(device)).detach().cpu().numpy()[:,:,0]
    print(Z.shape)
    #Z = icnn_couple.bwd_model(icnn_couple.fwd_model(torch.dstack((_X,_Y)).to(device))).detach().cpu().numpy()[:,:,1]
    # Plot the surface.
    #print(Z[0,0], Z[0,-1], Z[-1,0], Z[-1,-1], np.min(Z))

    #surf = ax.plot_trisurf(_X, _Y, Z, cmap=cm.coolwarm,
     #                   linewidth=0, antialiased=False, alpha=0.5)

    #fig.colorbar(surf, shrink=0.5, aspect=5)


    #fig.savefig('figs/simp_fwd')
    fig, ax = plt.subplots()
    plt.scatter(_X, _Y, c=Z)
    plt.colorbar()
    
    fig.savefig('figs/KL2_fwd')
    fig.savefig('figs/KL2_fwd.pdf')
    fig.savefig('figs/KL2_fwd.svg')
print(icnn_couple.stepsize)
          