# # relevant libraries
# import numpy as np
# import torch
# import torchvision
# import matplotlib.pyplot as plt
# from time import time
# from torchvision import datasets, transforms
# from torch import nn, optim
# import torch.nn.functional as F
# import random
# from joblib import dump, load
# from tqdm import tqdm
# from torch.utils.data import DataLoader, Sampler

# # for supressing warnings
# import warnings
# warnings.filterwarnings('ignore')

import wget
import tarfile

# defining subloaders for creating binary classification dataset from multi-class dataset
'''
MNIST dataset
'''
class subLoader(datasets.MNIST):
    def __init__(self, cls, **kwargs):
        super(subLoader, self).__init__(**kwargs)
        #self.targets and self.data are tensors
        self.mask = (self.targets.view(-1,1) == torch.tensor(cls).view(1,-1)).any(dim=1)
        
        self.targets = self.targets[self.mask]
        self.targets = (self.targets == cls[-1]).long()
        self.data = self.data[self.mask]

'''
FashionMNIST dataset
'''
class subLoader_FMNIST(datasets.FashionMNIST):
    def __init__(self, cls, **kwargs):
        super(subLoader_FMNIST, self).__init__(**kwargs)
        #self.targets and self.data are tensors
        self.mask = (self.targets.view(-1,1) == torch.tensor(cls).view(1,-1)).any(dim=1)
        
        self.targets = self.targets[self.mask]
        self.targets = (self.targets == cls[-1]).long()
        self.data = self.data[self.mask]

if dataset_name=='mnist':
  wget.download('www.di.ens.fr/~lelarge/MNIST.tar.gz')	
  #tar -zxvf MNIST.tar.gz  
  tf = tarfile.open("MNIST.tar.gz")
  tf.extractall()

  cls = [3,8] #MNIST  
  trainloader = DataLoader(subLoader(cls, root='./', train=True, download=True, transform=transforms.ToTensor()),**kwargs,shuffle=True))
  valloader = DataLoader(subLoader(cls, root='./', train=False, download=True, transform=transforms.ToTensor()),**kwargs,shuffle=False))  
  print('Loaded MNIST dataset!')
else:
  cls = [0,9] #FMNIST
  trainloader = DataLoader(subLoader_FMNIST(cls, root='./data/', train=True, download=True, transform=transforms.ToTensor()),**kwargs,shuffle=True)
  valloader = DataLoader(subLoader_FMNIST(cls, root='./data/', train=False, download=True, transform=transforms.ToTensor()),**kwargs,shuffle=False)
  print('Loaded FashionMNIST dataset!')


