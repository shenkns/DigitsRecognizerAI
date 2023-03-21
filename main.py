#%%
import torch
import torch.nn as nn
import torchvision as tv

import numpy as np
import matplotlib.pyplot as plt
import os


transform = tv.transforms.Compose([
    tv.transforms.ToTensor()
])

ds_mnist = tv.datasets.MNIST('./datasets', download=True, transform=transform)

plt.imshow(ds_mnist[0][0].numpy()[0])

print(ds_mnist)
