import torch
import torch.nn as nn
import torch.utils.data as Data 
import torchvision # this is the database of torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(4*4*64, 4*4*64),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Dropout(p=0.5),  # prevent overfitting
            nn.Linear(4*4*64, 86),
            nn.LeakyReLU(inplace=True, negative_slope=0.01)
        )
        
        self.last = nn.Linear(86,1)

    def forward(self, x):
        x = self.feature(x)
        # x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return_feature = self.classifier(x)
        x = self.last(return_feature)
        return x, return_feature