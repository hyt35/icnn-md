# Data-Driven MD with ICNNs
 
This is the code repository for the article "Data-Driven Mirror Descent with Input-Convex Neural Networks" by H.Y. Tan, S. Mukherjee, J. Tang, and C.-B. Schönlieb. An arXiv version of the article can be found [here](https://arxiv.org/abs/2206.06733).
<!---
![fwd_ratio](https://user-images.githubusercontent.com/56555137/184006090-10335507-ff1e-4bf7-86fc-33749716f4f4.png)![svm_margin_log_s](https://user-images.githubusercontent.com/56555137/184006293-6d08c1a7-2484-47dd-be05-ded926bc98bf.png)
--->
## Contents
This repository contains the following:
1. Code to train 1D and 2D LMD models;
2. Auxilliary NN used for SVM and NN training;
3. Code to create figures used in the article;
4. Some output figures

This code was run on Python 3.9.7 and Pytorch 1.11.0. Earlier versions may work but were not tested.

## Training and Evaluation
This section is split into three parts: "_simple_" containing the least-squares and SVM experiments with closed-form inverse in Section 4, "_artificial_" containing the SVM and linear classifier experiments in Section 5.1, and "_stl10_" containing the image denoising and inpainting experiments in Section 5.2 and 5.3.

### Closed-form Inverse MD
The 2d least-squares problem and the SVM training problem are implemented with both a quadratic form and one-hidden-layer neural network form of mirror potential. Those implemented using a one-layer neural network can be found in:
```2d_lsq.py, svm.py```,
and the one with quadratic form is 
```2d_lsq_MDLSQ.py, svm_MDLSQ.py```
The quadratic form is normally initialized near the identity, the ```xxx_MDLSQ_alt.py``` files allow for training with initializations around diagonal matrices with uniformly generated elements.

#### Training
Training is done by running:

```python xxx.py --train``

Optional arguments include number of training epochs ```--num_epochs```, number of batches after which to display ```--num_batches```, and checkpointing frequency ```--checkpoint_freq```. By default, checkpoints are saved into ```./checkpoints/xxx/n``` where n is the current epoch. Logs can be found in ```./logs/dd-mm_HHMM-xxx.log```, make sure to create the logs folder.

Testing is done by running

```python xxx.py --test --checkpoint=n```

where n is the checkpoint of the mirror descent model trained. This will produce loss comparison figures in ```./figs/xxx/n/```, with additional mirror potential visualization and mirror descent iteration visualization for the 2d_lsq case.


### SVM and Linear Classifier
