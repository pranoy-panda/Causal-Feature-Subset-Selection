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

# global variables
dataset_name = 'fmnist'
kwargs = {'batch_size':64, 'num_workers':2, 'pin_memory':True}