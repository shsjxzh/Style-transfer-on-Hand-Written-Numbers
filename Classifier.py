import torch
import torch.nn as nn
import torch.utils.data as Data 
import torchvision # this is the database of torch
from torchvision import datasets, transforms

class Classifier(torch.nn.Module):
    def __init__(self, num_feature, num_class):
        super(Classifier, self).__init__()
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
            nn.Linear(4*4*64, num_feature),
            nn.Dropout(p=0.5)
        )

        self.last = nn.Sequential(
            nn.Linear(num_feature, num_class)
        )

    def forward(self, x):
        x = self.feature(x)
        # x = self.conv2(x)
        x = x.view(x.size(0), -1)
        feature_vec = self.classifier(x)
        x = self.last(feature_vec)
        return x, feature_vec