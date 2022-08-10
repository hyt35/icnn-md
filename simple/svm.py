
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


checkpoint_path = 'checkpoints/svm/'
log_path = 'logs/%d-%m_%H%M-svm.log'
torch.cuda.set_device(1)



args=parse_import.parse_commandline_args()
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)  




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
        return torch.matmul(self.A, x[:,:,None])[...,0]

    def bwd_map(self, x):
        # clip for safety
        return torch.matmul(torch.inverse(self.A), x[:,:,None])[...,0]

    def zero_clip_weights(self): # do nothing
        #self.w.clamp(0.01)
        return self

    def scalar(self, x):
        return (torch.matmul(x,self.A)*x).sum(-1)/2

    def clamp_stepsizes(self):
        with torch.no_grad():
            self.stepsize.clamp_(self.ssmin,self.ssmax)
        return self


# load feature mapper
network = Net()
network.load_state_dict(torch.load('mnist_2layernet'))
pre_net = NetWithoutLast(network).to(device)
pre_net.eval()
del network


dim=51

icnn = MDNet_act(dim=dim, stepsize_init=0.1, num_iters = 10, stepsize_clamp = (0.05,0.5))
#icnn = MDNet_lsq(stepsize_init=0.1, num_iters = 10, stepsize_clamp = (0.05,0.5))
n_epochs = args.num_epochs
# noise_level = 0.05
# reg_param = 0.3
stepsize = 0.1
dataset_size = 1000 # number of random class pair to extract

bsize=1000
# closeness_reg = 1.0
# closeness_update_nepochs = 400
# closeness_update_multi = 1.05
#%%
loss_fn = torch.nn.MSELoss()
gen = torch.Generator()
gen.manual_seed(0) # deterministic W
#W = torch.Tensor([[[2,1],[1,2]]]).to(device)

if __name__ == "__main__":
    if args.train:
        mnist_data = torchvision.datasets.MNIST('/local/scratch/public/hyt35/datasets', train=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
        classes=(4,9)
        mnist_data_filtered_idx = (mnist_data.targets == classes[0]) | (mnist_data.targets == classes[1])
        mnist_data.data, mnist_data.targets = mnist_data.data[mnist_data_filtered_idx], mnist_data.targets[mnist_data_filtered_idx]

        logging.basicConfig(filename=datetime.now().strftime(log_path), filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
        logger = logging.getLogger()
        icnn.train()
        opt = torch.optim.Adam(icnn.parameters(),lr=1e-5,betas=(0.9,0.99))

        ind1 = classes[0]
        ind2 = classes[1]
        train_dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=dataset_size, shuffle=True)
        class_lossfn = torch.nn.MarginRankingLoss(reduction='sum', margin=1)

        #logger.info(W)
        for epoch in range(n_epochs):
            total_loss = 0
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

                init_wb = torch.randn(bsize, 51, requires_grad=True).to(device) # initialize bsize number of random svm initialization

                fwd = init_wb

                fwd.requires_grad_()
                loss = 0
                #closeness = icnn_couple.fwdbwdloss(torch.zeros_like(b).to(device))
                for stepsize in icnn.stepsize:
                    fwdGrad = svm_loss_grad(fwd)
                    fwd = icnn.bwd_map(icnn.fwd_map(fwd) - stepsize*fwdGrad) 
                    loss += svm_loss(fwd)


                #loss = svm_loss(fwd)

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
                print("curr avg loss", avg_loss)
                train_log = "epoch:[{}/{}] , avg_loss = {:.4f}".\
                    format(epoch+1, args.num_epochs, avg_loss)
                print(train_log)
                logger.info(train_log)
                total_loss = 0
            
            #print("Epoch", epoch, "total loss", total_loss, "fwdbwd", total_fwdbwd)
            # Checkpoint
            # Increase closeness regularization

            if (epoch%args.checkpoint_freq == args.checkpoint_freq-1):
                torch.save(icnn.state_dict(), checkpoint_path+str(epoch+1))
            # Log
                logger.info("\n====epoch:[{}/{}], epoch_loss = {:.2f}====\n".format(epoch+1, args.num_epochs, total_loss))
    else:
        # tester script
        mnist_data = torchvision.datasets.MNIST('/local/scratch/public/hyt35/datasets', train=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
        classes=(4,9)
        mnist_data_filtered_idx = (mnist_data.targets == classes[0]) | (mnist_data.targets == classes[1])
        mnist_data.data, mnist_data.targets = mnist_data.data[mnist_data_filtered_idx], mnist_data.targets[mnist_data_filtered_idx]

        train_dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=dataset_size, shuffle=True)
        class_lossfn = torch.nn.MarginRankingLoss(reduction='sum', margin=1)

        
        ind1 = classes[0]
        ind2 = classes[1]

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

            stepsize_scales = [1/4,1/2,1,2,4]

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
            ax_loss.set_ylabel('SVM hinge loss')

            fig_acc, ax_acc = plt.subplots(figsize = (10.5,7))
            plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.95)
            #fig_loss.suptitle("Test margin loss")
            #ax_acc.set_yscale('log')
            ax_acc.set_xlabel('Iterations')
            ax_acc.set_ylabel('Classification accuracy')

            ######
            initloss = svm_loss(fwd).item()
            initacc = classif_acc(fwd).item()

            mdloss = [initloss]
            mdacc = [initacc]
            avgss = torch.mean(icnn.stepsize).item()

            for ss in icnn.stepsize:
                #fwd = fwd.clamp(0,1)
                fwd.requires_grad_()
                fwdGrad0 = svm_loss_grad(fwd).detach().clone()
                fwd = icnn.bwd_map(icnn.fwd_map(fwd) - ss*fwdGrad0).detach()
                #fwd = icnn_bwd(icnn_fwd(fwd))
                #fwd_ = fwd.cpu().detach().numpy()
                #print("MD ", svm_loss(fwd).item())
                mdloss.append(svm_loss(fwd).item())
                mdacc.append(classif_acc(fwd).item())
                #print(svm_loss(fwd))

            # foofig, fooax = plt.subplots(figsize=(12,10))
            # bar = fwd.detach().cpu().numpy()
            # fooax.scatter(bar[:,0], bar[:,1])
            # foobar = torch.matmul(b,torch.inverse(W))[0].detach().cpu().numpy()
            # fooax.scatter(foobar[:,0], foobar[:,1],alpha=0.5)
            # foofig.savefig(fig_dir+'lsq_iters')

            for _ in range(10):
                fwd.requires_grad_()
                fwdGrad0 = svm_loss_grad(fwd).detach().clone()
                fwd = icnn.bwd_map(icnn.fwd_map(fwd) - avgss*fwdGrad0).detach()
                #fwd = icnn_bwd(icnn_fwd(fwd))
                #fwd_ = fwd.cpu().detach().numpy()
                #print("MD ext", svm_loss(fwd).item())
                mdloss.append(svm_loss(fwd).item())
                mdacc.append(classif_acc(fwd).item())
            ax_loss.plot(mdloss, label = "md adaptive", color='b', marker='o')
            ax_acc.plot(mdacc, label = "md adaptive", color='b', marker='o')


            max_mdloss = max(mdloss[:10])
            min_mdloss = min(mdloss)

            handles = [mlines.Line2D([], [], color='black', marker='', linestyle='None',label='Step-size multi')]

            for stepsize_scale, ms in zip(stepsize_scales, itertools.cycle('s^+*x')):
                torch.cuda.empty_cache()
                fwd = init_wb.to(device).requires_grad_()

                mdloss = [initloss]
                mdacc = [initacc]

                ## MD Fixed stepsize
                for i in range(20):
                    fwd.requires_grad_()
                    fwdGrad0 = svm_loss_grad(fwd).detach().clone()
                    fwd = icnn.bwd_map(icnn.fwd_map(fwd) - stepsize*stepsize_scale*fwdGrad0).detach()
                    #fwd = icnn_bwd(icnn_fwd(fwd))
                    fwd_ = fwd.cpu().detach().numpy()
                    #print("MD recon", svm_loss(fwd).item())
                    mdloss.append(svm_loss(fwd).item())
                    mdacc.append(classif_acc(fwd).item())


                gdloss = [initloss]
                gdacc = [initacc]
                ## GD
                fwd =  init_wb.to(device).requires_grad_()
                for i in range(20):
                    fwd.requires_grad_()
                    fwdGrad0 = svm_loss_grad(fwd)
                    fwd = fwd - stepsize*stepsize_scale*fwdGrad0
                    #print("GD recon", svm_loss(fwd).item())
                    gdloss.append(svm_loss(fwd).item())
                    gdacc.append(classif_acc(fwd).item())
                ## ADAM
                adamloss = []
                adamacc = []
                tmp =  init_wb.to(device)
                par = tmp.clone().detach()
                par.requires_grad_()
                optimizer = torch.optim.Adam([par], lr=stepsize_scale*0.05)
                for i in range(21):
                    optimizer.zero_grad()
                    loss = svm_loss(par)
                    adamloss.append(loss.item())
                    adamacc.append(classif_acc(par).item())
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    
                    #print("adam recon", svm_loss(par).item())

                    #fwd = (fwd-fwd.min())/(fwd.max()-fwd.min())

                ax_loss.plot(mdloss, label = "md " + str(stepsize_scale), color = 'm', marker=ms)
                ax_loss.plot(gdloss, label = "gd " + str(stepsize_scale), color = 'r', marker=ms)
                ax_loss.plot(adamloss, label = "adam " + str(stepsize_scale), color = 'g', marker=ms)

                ax_acc.plot(mdacc, label = "md " + str(stepsize_scale), color = 'm', marker=ms)
                ax_acc.plot(gdacc, label = "gd " + str(stepsize_scale), color = 'r', marker=ms)
                ax_acc.plot(adamacc, label = "adam " + str(stepsize_scale), color = 'g', marker=ms)
                handles.append(mlines.Line2D([], [], color='black', marker=ms, linestyle='None',
                                    markersize=10, label="{:.2f}".format(stepsize_scale)))
                
            #ax_loss.set_ylim((min_mdloss/5,max_mdloss*10))
            #plt.ylim((min_mdloss/5,min_mdloss*10))


            #ax_loss.set_xticks(np.arange(0,21,step=5))


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
            #'>^+*x'

            # _mrkempty =  mlines.Line2D([], [], color='black', marker='', linestyle='None',label='Step-size multi')
            # _mrk1 = mlines.Line2D([], [], color='black', marker='>', linestyle='None',
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
            fig_loss.savefig(fig_dir+'lsq_loss.pdf',bbox_inches='tight')
            fig_loss.savefig(fig_dir+'lsq_loss.svg',bbox_inches='tight')

            box = ax_acc.get_position()
            ax_acc.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            legend1 = ax_acc.legend(handles=[md_adap_patch,md_patch,gd_adap_patch,adam_adap_patch], bbox_to_anchor=(1.02, 0.7),
                                    loc='center left', borderaxespad=0.)
            #[1/2,1,2,4,10]
            #'>^+*x'

            # _mrkempty =  mlines.Line2D([], [], color='black', marker='', linestyle='None',label='Step-size multi')
            # _mrk1 = mlines.Line2D([], [], color='black', marker='>', linestyle='None',
            #                         markersize=10, label='1/4')
            # _mrk2 = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
            #                         markersize=10, label='1/2')
            # _mrk3 = mlines.Line2D([], [], color='black', marker='+', linestyle='None',
            #                         markersize=10, label='1')
            # _mrk4 = mlines.Line2D([], [], color='black', marker='*', linestyle='None',
            #                           markersize=10, label='2')
            # _mrk5 = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
            #                           markersize=10, label='4')

            ax_acc.legend(handles=handles, bbox_to_anchor=(1.02,0.3),
                                    loc='center left', borderaxespad=0.)
            ax_acc.add_artist(legend1)

            #ax_acc.legend(loc='upper right', fontsize='x-small')
            #ax_prob.legend(loc='lower right', fontsize='x-small')
            fig_acc.savefig(fig_dir+'lsq_acc',bbox_inches='tight')
            fig_acc.savefig(fig_dir+'lsq_acc.svg',bbox_inches='tight')
            fig_acc.savefig(fig_dir+'lsq_acc.pdf',bbox_inches='tight')
            #plt.show()
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            X = np.arange(-2, 2, 0.05)
            Y = np.arange(-2, 2, 0.05)
            X, Y = np.meshgrid(X, Y)

            #foo = X**2+Y**2 +1e-3
            #foo = (2*X+Y)**2+(X+2*Y)**2+1e-1
            _X = torch.Tensor(X)
            _Y = torch.Tensor(Y)
            print(_X.shape)
            print(torch.dstack((_X,_Y)).shape)
            Z = icnn.scalar(torch.dstack((_X,_Y, torch.zeros_like(_X[:,:,None]).repeat(1,1,49))).reshape(-1,51).to(device)).reshape(80,80).detach().cpu().numpy()
            #Z = icnn_couple.bwd_map(icnn_couple.fwd_map(torch.dstack((_X,_Y)).to(device))).detach().cpu().numpy()[:,:,1]
            # Plot the surface.
            #print(Z[0,0], Z[0,-1], Z[-1,0], Z[-1,-1], np.min(Z))

            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)

            fig.colorbar(surf, shrink=0.5, aspect=5)

            fig.savefig(fig_dir+'lsq_fwd',bbox_inches='tight')
            fig.savefig(fig_dir+'lsq_fwd.pdf',bbox_inches='tight')
            plt.show()
            print(icnn.stepsize)
            print(icnn.A)
            print(torch.eig(icnn.A)[0])
            print(icnn.w)