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

# for supressing warnings
import warnings
warnings.filterwarnings('ignore')


def test_basemodel(valloader,bb_model):
  # testing the black box model performance on the entire validation dataset
  correct_count, all_count = 0, 0
  for images,labels in valloader:
    for i in range(len(labels)):
      img = images[i]
      img = img.unsqueeze(0)
      with torch.no_grad():
          out = bb_model(img)

      pred_label = torch.argmax(out)
      true_label = labels.numpy()[i]
      if(true_label == pred_label):
        correct_count += 1
      all_count += 1

  print("Number Of Images Tested =", all_count)
  print("\nModel Accuracy =", (correct_count/all_count)) 