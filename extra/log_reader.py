import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, AutoLocator

# plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
# plt.rc('legend', fontsize=14) 
# https://www.reddit.com/r/learnpython/comments/1hs2aw/parsing_a_log_file_with_numpy/
# https://www.delftstack.com/howto/python/python-log-parser/
plt.rc('axes', labelsize=16)
start=time.time()

# file_names = ["logs/11-05_15:00-lsq.log"] # change reg update freq to 400 and starterct to 200
# starter_ct = 200
# reg_update_freq = 400

file_names = ["logs/09-05_17:09-nn.log",
"logs/09-05_17:09-nn_skip.log",
"logs/09-05_17:09-svm.log",
"../ICNN-STL10/logs/27-04_15:16-denoise.log",
"../ICNN-STL10/logs/27-04_13:09-inpainting-denoising.log"]
starter_ct = 50
reg_update_freq = 50
for file_name in file_names:


    # NN, NN_SKIP, SVM, DENOISE, INPAINT: 50
    # LSQ: 400

    file = open(file_name, "r")
    epochs = []
    losses = []
    fwdbwd = []

    for line in file.readlines():
        #print(len(line))
        if (len(line)<14) or (line[14] != "e"): # discard mass display
            continue
        line = line[21:] # discard everything before epoch number
        epochs.append(line.split('/')[0]) # epoch number
        _, loss_string, fwdbwd_string = line.split(',')

        losses.append(loss_string[12:])
        fwdbwd.append(fwdbwd_string[14:-1])

    epochs = np.asarray(epochs, dtype=int)
    losses = np.asarray(losses, dtype=float)
    fwdbwd = np.asarray(fwdbwd, dtype=float)

    epoch_multi_divisor = 1.05**np.floor((epochs-1)/reg_update_freq)

    obj_losses = losses - epoch_multi_divisor*fwdbwd

    #print(len(epochs))

    # fig, ax = plt.subplots(2,1, figsize=(9,7))


    # ax[0].plot(epochs[epochs>=starter_ct], obj_losses[epochs>=starter_ct])
    # ax[1].plot(epochs[epochs>=starter_ct], fwdbwd[epochs>=starter_ct])
    # #ax[0].set_yscale('log')
    # ax[1].set_yscale('log')
    # ax[0].set_title('Objective loss')
    # ax[1].set_title('Forward-Backward error')
    # #print(epochs[np.logical_and(epochs%reg_update_freq==0, epochs>=starter_ct)])
    # #ax[0].set_xticks(epochs[np.logical_and(epochs%reg_update_freq==0, epochs>=starter_ct)]) # set ticks where regularization parameter changes
    # #ax[1].set_xticks(epochs[np.logical_and(epochs%reg_update_freq==0, epochs>=starter_ct)])
    # ax[0].grid(visible=True, which='major', axis='x')
    # ax[1].grid(visible=True, which='major', axis='x')
    # #ax[0].set_xticklabels([])
    # #ax[1].set_xticklabels([])


    # reg_update_epoch = epochs[np.logical_and(epochs%reg_update_freq==0, epochs>=starter_ct)]

    # ax[0].xaxis.set_major_locator(FixedLocator(reg_update_epoch[::20]))
    # ax[1].xaxis.set_major_locator(FixedLocator(reg_update_epoch[::20]))

    # ax[0].xaxis.set_minor_locator(FixedLocator(epochs[np.logical_and(epochs%reg_update_freq==0, epochs>=starter_ct)]))
    # ax[1].xaxis.set_minor_locator(FixedLocator(epochs[np.logical_and(epochs%reg_update_freq==0, epochs>=starter_ct)]))


    # #ax[0].grid(visible=False, which='major', axis='x')
    # #ax[1].grid(visible=False, which='major', axis='x')
    # ax[0].grid(visible=True, which='minor', axis='x')
    # ax[1].grid(visible=True, which='minor', axis='x')

    # #fig.suptitle(file_name.split('/')[-1].split('-')[2].split('.')[0])
    # file_name = file_name.replace(":","")
    # fig.savefig("log_plot/"+file_name.split('/')[-1][:-4])
    # fig.savefig("log_plot/"+file_name.split('/')[-1][:-4]+".svg")
    # fig.savefig("log_plot/"+file_name.split('/')[-1][:-4]+".pdf")
    # end=time.time()
    # print("Elapsed time: ", end-start)

    # #complete = np.vstack((epochs,losses,fwdbwd))
    # #print(complete)

    fig, ax1 = plt.subplots(figsize=(9,7))
    ax2 = ax1.twinx()

    ax1.plot(epochs[epochs>=starter_ct], obj_losses[epochs>=starter_ct], alpha=0.5, color='b')
    ax2.plot(epochs[epochs>=starter_ct], fwdbwd[epochs>=starter_ct], color='r')
    #ax[0].set_yscale('log')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.set_ylabel('Objective loss', color='b')
    ax2.set_ylabel('Forward-backward error', color = 'r')
    #print(epochs[np.logical_and(epochs%reg_update_freq==0, epochs>=starter_ct)])
    #ax[0].set_xticks(epochs[np.logical_and(epochs%reg_update_freq==0, epochs>=starter_ct)]) # set ticks where regularization parameter changes
    #ax[1].set_xticks(epochs[np.logical_and(epochs%reg_update_freq==0, epochs>=starter_ct)])
    ax1.grid(visible=True, which='major', axis='x')

    #ax[0].set_xticklabels([])
    #ax[1].set_xticklabels([])


    reg_update_epoch = epochs[np.logical_and(epochs%reg_update_freq==0, epochs>=starter_ct)]

    ax1.xaxis.set_major_locator(FixedLocator(reg_update_epoch[::20]))
    #ax[1].xaxis.set_major_locator(FixedLocator(reg_update_epoch[::20]))

    ax1.xaxis.set_minor_locator(FixedLocator(epochs[np.logical_and(epochs%reg_update_freq==0, epochs>=starter_ct)]))
    #ax[1].xaxis.set_minor_locator(FixedLocator(epochs[np.logical_and(epochs%reg_update_freq==0, epochs>=starter_ct)]))


    #ax[0].grid(visible=False, which='major', axis='x')
    #ax[1].grid(visible=False, which='major', axis='x')
    ax1.grid(visible=True, which='minor', axis='x')
    #ax[1].grid(visible=True, which='minor', axis='x')

    #fig.suptitle(file_name.split('/')[-1].split('-')[2].split('.')[0])
    file_name = file_name.replace(":","")
    fig.savefig("log_plot_combine/"+file_name.split('/')[-1][:-4])
    fig.savefig("log_plot_combine/"+file_name.split('/')[-1][:-4]+".svg")
    fig.savefig("log_plot_combine/"+file_name.split('/')[-1][:-4]+".pdf")
    end=time.time()
    print("Elapsed time: ", end-start)

    #complete = np.vstack((epochs,losses,fwdbwd))
    #print(complete)