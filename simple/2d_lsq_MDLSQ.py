
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
from pathlib import Path
import re
import itertools
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib import cm


checkpoint_path = 'checkpoints/lsq_mdlsq2/'
torch.cuda.set_device(1)

plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)  

args=parse_import.parse_commandline_args()

Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

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

class MDNet_act(nn.Module):
    # this guy will have forward w^T \sigma(Ax)
    # where sigma is a smooth approx to leaky relu
    # given by alpha + (1-alpha) log (1+exp(x))
    # this approximates leaky relu with neg slope alpha (taken 0.2)

    def __init__(self, dim = 10, stepsize_init = 0.01, num_iters = 10, stepsize_clamp = (0.001,0.1)):
        super(MDNet_act, self).__init__()
        self.dim = dim # input dimension of data
        self.stepsize = nn.Parameter(stepsize_init * torch.ones(num_iters).to(device))
        self.num_iters = num_iters
        self.ssmin = stepsize_clamp[0]
        self.ssmax = stepsize_clamp[1]
        self.w = nn.Parameter(torch.rand(dim).to(device)/dim)
        self.A = nn.Parameter((torch.eye(dim)+0.01*torch.randn(dim,dim)).to(device))

        #self.w.weight.data.normal_(-1,1).exp_()
        #slope of leaky-relu
        self.alpha = 0.2 

    def fwd_map(self, x):
        # x in (batch, dim)
        Ax = torch.matmul(self.A, x[:,:,None])[...,0] # batch multi, maybe better way to do this
        return self.alpha*torch.matmul(self.A.T,self.w) + (1-self.alpha)*self.w*torch.exp(Ax)/(1+torch.exp(Ax))

    def bwd_map(self, x):
        # clip for safety
        y = torch.reciprocal((1-self.alpha)*self.w)*(x-self.alpha*torch.matmul(self.A.T,self.w))
        y.clamp_(0.01,0.99)
        z = torch.matmul(torch.inverse(self.A), torch.log(y/(1-y))[:,:,None])[...,0]
        return z

    def zero_clip_weights(self):
        self.w.clamp(0.01)
        return self

    def scalar(self, x):
        Ax = torch.matmul(x, self.A.T)
        return (self.w*(self.alpha*Ax+(1-self.alpha)*torch.log(1+torch.exp(Ax)))).sum(-1)

    def clamp_stepsizes(self):
        with torch.no_grad():
            self.stepsize.clamp_(self.ssmin,self.ssmax)
        return self

class MDNet_lsq(nn.Module):
    def __init__(self, dim = 10, stepsize_init = 0.01, num_iters = 10, stepsize_clamp = (0.001,0.1)):
        super(MDNet_lsq, self).__init__()
        self.dim = dim # input dimension of data
        self.stepsize = nn.Parameter(stepsize_init * torch.ones(num_iters).to(device))
        self.num_iters = num_iters
        self.ssmin = stepsize_clamp[0]
        self.ssmax = stepsize_clamp[1]
        self.A = nn.Parameter((torch.eye(dim)+0.01*torch.randn(dim,dim))).to(device)

        #self.w.weight.data.normal_(-1,1).exp_()
        #slope of leaky-relu
        self.alpha = 0.2 

    def fwd_map(self, x):
        # x in (batch, dim)
        return torch.matmul((self.A+self.A.T)/2, x[:,:,None])[...,0]

    def bwd_map(self, x):
        # clip for safety
        return torch.matmul(torch.inverse((self.A+self.A.T)/2), x[:,:,None])[...,0]

    def zero_clip_weights(self): # do nothing
        #self.w.clamp(0.01)
        return self

    def scalar(self, x):
        return (torch.matmul(x,self.A)*x).sum(-1)/2

    def clamp_stepsizes(self):
        with torch.no_grad():
            self.stepsize.clamp_(self.ssmin,self.ssmax)
        return self

dim=2

icnn = MDNet_lsq(dim=2, stepsize_init=0.1, num_iters = 10, stepsize_clamp = (0.05,0.5))
#icnn = MDNet_lsq(stepsize_init=0.1, num_iters = 10, stepsize_clamp = (0.05,0.5))
n_epochs = args.num_epochs
# noise_level = 0.05
# reg_param = 0.3
stepsize = 0.1
bsize=1000
# closeness_reg = 1.0
# closeness_update_nepochs = 400
# closeness_update_multi = 1.05
#%%
loss_fn = torch.nn.MSELoss()
gen = torch.Generator()
gen.manual_seed(0) # deterministic W
W = torch.Tensor([[[2,1],[1,2]]]).to(device)

if __name__ == "__main__":
    if args.train:
        logging.basicConfig(filename=datetime.now().strftime('logs/%d-%m_%H%M-lsq-mdlsq.log'), filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
        logger = logging.getLogger()
        icnn.train()
        opt = torch.optim.Adam(icnn.parameters(),lr=1e-4,betas=(0.9,0.99))
        total_loss = 0
        total_closeness = 0


        logger.info(W)
        for epoch in range(n_epochs):

            x = torch.randn(bsize, dim).to(device)
            
            b = torch.randn(bsize,dim).to(device)

            # define objective functions
            # min ||Wx-b||_2^2
            def recon_err(x):
                return torch.sum((torch.matmul(x,W)[0] - b)**2)
            
            def recon_err_grad(x):
                return autograd.grad(recon_err(x), x)[0]


            fwd = x.to(device)
            fwd.requires_grad_()
            loss = 0
            #closeness = icnn_couple.fwdbwdloss(torch.zeros_like(b).to(device))
            for stepsize in icnn.stepsize:
                fwdGrad = recon_err_grad(fwd)
                fwd = icnn.bwd_map(icnn.fwd_map(fwd) - stepsize*fwdGrad) 
                loss += recon_err(fwd)


            #loss = recon_err(fwd)

            err = loss
            
            opt.zero_grad()
            err.backward()
            opt.step()
            
            icnn.zero_clip_weights()
            #icnn.clip_bwd()
            #icnn_bwd.zero_clip_weights()
            icnn.clamp_stepsizes()
            
            total_loss += err.item()

            if(epoch % args.num_batches == args.num_batches-1):
                avg_loss = total_loss/args.num_batches/bsize
                avg_fwdbwd = total_closeness/args.num_batches/bsize
                print("curr avg loss", avg_loss)
                train_log = "epoch:[{}/{}] , avg_loss = {:.4f}".\
                    format(epoch+1, args.num_epochs, avg_loss)
                print(train_log)
                logger.info(train_log)
                total_loss = 0
                total_closeness = 0
            
            #print("Epoch", epoch, "total loss", total_loss, "fwdbwd", total_fwdbwd)
            # Checkpoint
            # Increase closeness regularization

            if (epoch%args.checkpoint_freq == args.checkpoint_freq-1):
                torch.save(icnn.state_dict(), checkpoint_path+str(epoch+1))
            # Log
                logger.info("\n====epoch:[{}/{}], epoch_loss = {:.2f}====\n".format(epoch+1, args.num_epochs, total_loss))
    else:
        # tester script

        x = torch.randn(bsize, dim).to(device)
        b = torch.randn(bsize,dim).to(device)
        # change these loss functions
        def recon_err(x):
            return torch.sum((torch.matmul(x,W)[0] - b)**2)

        def recon_err_grad(x):
            return autograd.grad(recon_err(x), x)[0]
        stepsize_scales = [1/4,1/2,1]

        ###########
        # for printing
        read_checkpoint = checkpoint_path+str(args.checkpoint)
        icnn.load_state_dict(torch.load(read_checkpoint))
        fig_dir = "figs"+re.sub(r'^.*?/', '/',read_checkpoint)+"/"
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        #print(fig_dir)


        fig_loss, ax_loss = plt.subplots(figsize = (10.5,7))
        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.95)
        #fig_loss.suptitle("Test margin loss")
        ax_loss.set_yscale('log')
        ax_loss.set_xlabel('Iterations')
        ax_loss.set_ylabel('Least square loss')



        ######
        fwd = x.to(device)


        initloss = recon_err(fwd).item()

        mdloss = [initloss]

        avgss = torch.min(icnn.stepsize).item()

        for ss in icnn.stepsize:
            #fwd = fwd.clamp(0,1)
            fwd.requires_grad_()
            fwdGrad0 = recon_err_grad(fwd).detach().clone()
            fwd = icnn.bwd_map(icnn.fwd_map(fwd) - ss*fwdGrad0).detach()
            #fwd = icnn_bwd(icnn_fwd(fwd))
            #fwd_ = fwd.cpu().detach().numpy()
            #print("MD ", svm_loss(fwd).item())
            mdloss.append(recon_err(fwd).item())
            #print(recon_err(fwd))

        foofig, fooax = plt.subplots(figsize=(12,10))
        bar = fwd.detach().cpu().numpy()
        fooax.scatter(bar[:,0], bar[:,1])
        foobar = torch.matmul(b,torch.inverse(W))[0].detach().cpu().numpy()
        fooax.scatter(foobar[:,0], foobar[:,1],alpha=0.5)
        foofig.savefig(fig_dir+'lsq_iters',bbox_inches='tight')
        foofig.savefig(fig_dir+'lsq_iters.svg',bbox_inches='tight')
        foofig.savefig(fig_dir+'lsq_iters.pdf',bbox_inches='tight')
        for _ in range(10):
            fwd.requires_grad_()
            fwdGrad0 = recon_err_grad(fwd).detach().clone()
            fwd = icnn.bwd_map(icnn.fwd_map(fwd) - avgss*fwdGrad0).detach()
            #fwd = icnn_bwd(icnn_fwd(fwd))
            #fwd_ = fwd.cpu().detach().numpy()
            #print("MD ext", recon_err(fwd).item())
            mdloss.append(recon_err(fwd).item())

        ax_loss.plot(mdloss, label = "md adaptive", color='b', marker='o')



        max_mdloss = max(mdloss[:10])
        min_mdloss = min(mdloss)

        handles = [mlines.Line2D([], [], color='black', marker='', linestyle='None',label='Step-size multi')]

        for stepsize_scale, ms in zip(stepsize_scales, itertools.cycle('s^+*x')):
            torch.cuda.empty_cache()
            fwd = x.to(device).requires_grad_()

            mdloss = [initloss]

            ## MD Fixed stepsize
            for i in range(20):
                fwd.requires_grad_()
                fwdGrad0 = recon_err_grad(fwd).detach().clone()
                fwd = icnn.bwd_map(icnn.fwd_map(fwd) - stepsize*stepsize_scale*fwdGrad0).detach()
                #fwd = icnn_bwd(icnn_fwd(fwd))
                fwd_ = fwd.cpu().detach().numpy()
                #print("MD recon", recon_err(fwd).item())
                mdloss.append(recon_err(fwd).item())



            gdloss = [initloss]

            ## GD
            fwd =  x.to(device).requires_grad_()
            for i in range(20):
                fwd.requires_grad_()
                fwdGrad0 = recon_err_grad(fwd)
                fwd = fwd - stepsize*stepsize_scale*fwdGrad0
                #print("GD recon", recon_err(fwd).item())
                gdloss.append(recon_err(fwd).item())

            ## ADAM
            adamloss = []
            tmp =  x.to(device)
            par = tmp.clone().detach()
            par.requires_grad_()
            optimizer = torch.optim.Adam([par], lr=stepsize_scale*0.05)
            for i in range(21):
                optimizer.zero_grad()
                loss = recon_err(par)
                adamloss.append(loss.item())

                loss.backward(retain_graph=True)
                optimizer.step()
                
                #print("adam recon", recon_err(par).item())

                #fwd = (fwd-fwd.min())/(fwd.max()-fwd.min())

            ax_loss.plot(mdloss, label = "md " + str(stepsize_scale), color = 'm', marker=ms)
            ax_loss.plot(gdloss, label = "gd " + str(stepsize_scale), color = 'r', marker=ms)
            ax_loss.plot(adamloss, label = "adam " + str(stepsize_scale), color = 'g', marker=ms)

            handles.append(mlines.Line2D([], [], color='black', marker=ms, linestyle='None',
                                markersize=10, label="{:.2f}".format(stepsize_scale)))
            
        #ax_loss.set_ylim((min_mdloss/5,max_mdloss*10))
        #plt.ylim((min_mdloss/5,min_mdloss*10))


        ax_loss.set_xticks(np.arange(0,21,step=5))


        md_adap_patch = mpatches.Patch(color='b', label='Adaptive LMD')
        md_patch = mpatches.Patch(color='m', label='LMD')
        gd_adap_patch = mpatches.Patch(color='r', label='GD')
        adam_adap_patch = mpatches.Patch(color='g', label='Adam')

        box = ax_loss.get_position()
        ax_loss.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        legend1 = ax_loss.legend(handles=[md_adap_patch,md_patch,gd_adap_patch,adam_adap_patch], bbox_to_anchor=(1.02, 0.7),
                                loc='center left', borderaxespad=0.)
        #[1/2,1,2,4,10]
        #'s^+*x'

        # _mrkempty =  mlines.Line2D([], [], color='black', marker='', linestyle='None',label='Step-size multi')
        # _mrk1 = mlines.Line2D([], [], color='black', marker='s', linestyle='None',
        #                         markersize=10, label='1/4')
        # _mrk2 = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
        #                         markersize=10, label='1/2')
        # _mrk3 = mlines.Line2D([], [], color='black', marker='+', linestyle='None',
        #                         markersize=10, label='1')
        # _mrk4 = mlines.Line2D([], [], color='black', marker='*', linestyle='None',
        #                           markersize=10, label='2')
        # _mrk5 = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
        #                           markersize=10, label='4')

        ax_loss.legend(handles=handles, bbox_to_anchor=(1.02,0.3),
                                loc='center left', borderaxespad=0.)
        ax_loss.add_artist(legend1)

        #ax_loss.legend(loc='upper right', fontsize='x-small')
        #ax_prob.legend(loc='lower right', fontsize='x-small')
        fig_loss.savefig(fig_dir+'lsq_loss',bbox_inches='tight')
        fig_loss.savefig(fig_dir+'lsq_loss.svg',bbox_inches='tight')
        fig_loss.savefig(fig_dir+'lsq_loss.pdf',bbox_inches='tight')
        plt.show()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X = np.arange(-2, 2, 0.05)
        Y = np.arange(-2, 2, 0.05)
        X, Y = np.meshgrid(X, Y)

        #foo = X**2+Y**2 +1e-3
        #foo = (2*X+Y)**2+(X+2*Y)**2+1e-1
        _X = torch.Tensor(X)
        _Y = torch.Tensor(Y)
        Z = icnn.scalar(torch.dstack((_X,_Y)).to(device)).detach().cpu().numpy()
        #Z = icnn_couple.bwd_map(icnn_couple.fwd_map(torch.dstack((_X,_Y)).to(device))).detach().cpu().numpy()[:,:,1]
        # Plot the surface.
        #print(Z[0,0], Z[0,-1], Z[-1,0], Z[-1,-1], np.min(Z))

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)

        fig.colorbar(surf, shrink=0.5, aspect=5)

        fig.savefig(fig_dir+'lsq_fwd',bbox_inches='tight')
        fig.savefig(fig_dir+'lsq_fwd.svg',bbox_inches='tight')
        fig.savefig(fig_dir+'lsq_fwd.pdf',bbox_inches='tight')
        plt.show()
        print(icnn.stepsize)
        print(icnn.A)
        #print(icnn.w)