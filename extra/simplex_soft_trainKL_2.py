
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
from torch.nn import KLDivLoss
device='cuda'

checkpoint_path = '/local/scratch/public/hyt35/ICNN-MD/ICNN-ARTIFICIAL/checkpoints/simplex_KL2/'
logging.basicConfig(filename=datetime.now().strftime('logs/%d-%m_%H:%M-simplex_KL2.log'), filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
lossfn = KLDivLoss(reduction='sum', log_target=False)
logger = logging.getLogger()
torch.cuda.set_device(1)
args=parse_import.parse_commandline_args()

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


dim=3

icnn_couple = ICNNCouple(stepsize_init = 0.1, num_iters = 10, stepsize_clamp = (0.01,0.5))
icnn_couple.init_fwd(in_dim = dim, hidden = 30, num_layers=4, strong_convexity = 0.2)
icnn_couple.fwd_model.initialize_weights()
icnn_couple.init_bwd(in_dim = dim, hidden = 40, num_layers=4, strong_convexity = 0.2)

n_epochs = args.num_epochs
noise_level = 0.05
reg_param = 0.3
stepsize = 0.01
bsize=2000
closeness_reg = 1.0
closeness_update_nepochs = 400
closeness_update_multi = 1.05
#%%
loss_fn = torch.nn.MSELoss()
gen = torch.Generator()
gen.manual_seed(0) # deterministic W


icnn_couple.train()
opt = torch.optim.Adam(icnn_couple.parameters(),lr=1e-4,betas=(0.9,0.99))
total_loss = 0
total_closeness = 0
# 2D simplex
# x1,x2>0, x1+x2<1 (x3=1-x1-x2)
#W = torch.Tensor([[[2,1],[1,2]]]).to(device)
# Uniform sampler on simplex
expSampler = torch.distributions.exponential.Exponential(torch.ones(bsize, 3))

#logger.info(W)
# https://arxiv.org/pdf/1101.6081.pdf

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

for epoch in range(n_epochs):

    _x = expSampler.sample()
    _x = _x/_x.sum(1)[:,None] # so when projected, get uniform sample on simplex.
    x = _x.to(device)
    #x = simplexProjection(x) # only used for testing
    # b = torch.rand(bsize,3).to(device) # hypercube [0,1]^3 uniform
    # b_simplex = simplexProjection(b)
    _b = expSampler.sample()
    _b = _b/_b.sum(1)[:,None] # so when projected, get uniform sample on simplex.
    b = _b.to(device)
    #b_simplex = simplexProjection(b) # only used for testing

    # define objective functions
    # these are parameterized by b
    # 
    def obj(x):
        #return lossfn(torch.log(x), b)
        return lossfn(torch.log(b), x.clamp(0.001))
    def obj_grad(x):
        return autograd.grad(obj(x), x)[0]
    def off_simplex_loss(x):
        # this can be done using projections but that is slow. 
        # so we consider the projection to the hyperplane sum x_i = 1
        # and to the positive orthant
        # first term is abs sum of negative component
        # second term is distance to hyperplane
        return (x.clamp(max=0)**2).sum() + (x.sum(1)-1).abs().sum()
        #return ((x-simplexProjection(x))**2).sum()



    fwd = x.to(device)
    fwd.requires_grad_()
    loss = 0
    #closeness = icnn_couple.fwdbwdloss(torch.zeros_like(b).to(device))
    closeness = 0
    simplex_loss = 0
    for stepsize in icnn_couple.stepsize:
        fwdGrad = obj_grad(fwd)
        
        fwd = icnn_couple.bwd_model(icnn_couple.fwd_model(fwd) - stepsize*fwdGrad)
        loss += obj(fwd) 
        closeness += icnn_couple.fwdbwdloss(fwd)
        simplex_loss += off_simplex_loss(fwd)
    #loss = recon_err(fwd)
    closeness += icnn_couple.fwdbwdloss(x)*10 # heavily enforce consistency on simplex

    err = loss+(closeness)*closeness_reg+(simplex_loss)*10
    
    opt.zero_grad()
    err.backward()
    opt.step()
    
    icnn_couple.clip_fwd()
    #icnn_couple.clip_bwd()
    #icnn_bwd.zero_clip_weights()
    icnn_couple.clamp_stepsizes()
    
    total_loss += err.item()
    total_closeness += closeness.item()

    if(epoch % args.num_batches == args.num_batches-1):
        avg_loss = total_loss/args.num_batches/bsize
        avg_fwdbwd = total_closeness/args.num_batches/bsize
        print("curr loss", loss.item(), "curr closeness", closeness.item(), "curr simplex loss", simplex_loss.item())
        train_log = "epoch:[{}/{}] , avg_loss = {:.4f}, avg_fwdbwd = {:.4f}, simplex_loss = {:.3f}".\
            format(epoch+1, args.num_epochs, avg_loss, avg_fwdbwd, simplex_loss.item())
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
