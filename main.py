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

# importing local libraries
from config import *
from dataset_file import *
from utils import *
from models import *

num_epochs = 4
lr = 0.001
batch_size = kwargs['batch_size']
bb_model = BaseModel()

LossFunc = torch.nn.CrossEntropyLoss(size_average = True)
optimizer = torch.optim.Adam(bb_model.parameters(),lr = lr) 

#training loop
for epoch in range(num_epochs):
  with tqdm(trainloader, unit="batch") as tepoch:
    for data, target in tepoch:
      tepoch.set_description("Epoch "+str(epoch))
      
      #data, target = data.to(device), target.to(device)
      optimizer.zero_grad()

      outputs = bb_model(data)
      loss = LossFunc(outputs, target)

      predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
      correct = (predictions == target).sum().item()
      accuracy = correct / batch_size
      
      loss.backward()
      optimizer.step()

      tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

# uncomment to save the model
torch.save(bb_model, 'mnist_model.pt')    

# testing the model on held-out validation dataset
test_basemodel(valloader,bb_model)

