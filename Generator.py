import torch
import torch.nn as nn
import torch.utils.data as Data 
import torchvision # this is the database of torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.pre_gen = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=1024),
            nn.Linear(1024, 7*7*128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=7*7*128),
        )
        self.gen = nn.Sequential(
            # Unflatten(batch_size, 128, 7, 7),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
    def forward(self, x):
        batch_size = x.size(0)
        x = self.pre_gen(x)
        x = x.view(batch_size, 128, 7, 7)
        x = self.gen(x)
        return x