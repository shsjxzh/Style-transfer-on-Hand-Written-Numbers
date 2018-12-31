import torch
import torch.nn as nn
import torch.utils.data as Data 
import torchvision # this is the database of torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

class A_Encoder(nn.Module):
    def __init__(self, num_feature, use_gpu=True):
        super(A_Encoder, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # self.log_var = nn.Sequential(nn.Linear(128 * 14 * 14, 1024))
        self.my_mean = nn.Sequential(nn.Linear(4* 4 *64, num_feature))
        self.use_gpu = use_gpu
    
    def encode(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return self.my_mean(x) # , self.log_var(x) 

    def reparametrize(self, my_mean):       
        eps = torch.normal(torch.zeros_like(my_mean), 1)
        if self.use_gpu:
            device = torch.device("cuda:" + str(my_mean.get_device()))
        else:
            device = torch.device("cpu")

        eps = eps.to(device)

        return eps + my_mean

    def forward(self, x):
        my_mean = self.encode(x)
        z = self.reparametrize(my_mean)
        return z, my_mean