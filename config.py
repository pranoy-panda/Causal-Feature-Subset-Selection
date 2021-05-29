# relevant libraries
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import random
#from joblib import dump, load
from tqdm import tqdm
from torch.utils.data import DataLoader, Sampler
import sys

# for supressing warnings
import warnings
warnings.filterwarnings('ignore')


# set the seeds for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
SEED = 12345
random.seed(SEED) 
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# global variables for basemodel
dataset_name = 'mnist'
kwargs = {'batch_size':64, 'num_workers':2, 'pin_memory':True}
num_epochs_basemodel = 1
lr_basemodel = 0.001 

# global vars for feature selection algorithm (tau, k, etc)
# M*N x M*N is the size of the image
M = 7 # selection map size(assuming a square shaped selection map) 
N = 4 # patch size(square patch)
num_patches = 6# number of patches to be selected 
k = M*M-num_patches# number of patches for S_bar
tau = 0.1
num_epochs = 1
lr = 0.0001
best_val_acc = 0
num_init = 1 # number of initializations of the explainer
val_acc_list = []
ice_list = []
