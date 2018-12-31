import torch
import torch.nn as nn
import torch.utils.data as Data 
import torchvision # this is the database of torch
import torchvision.models as models
from torchvision.utils import save_image
from torch.nn import init

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

import os
import numpy as np
from torchvision import datasets, transforms

# Hyper Parameters
EPOCH = 200                    # the training times
BATCH_SIZE = 128               # not use all data to train
SHOW_STEP = 101                # show the result after how many steps
CHANGE_EPOCH = 4               # change learning rate
SAVE_EPOCH = 10
USE_GPU = True                 # CHANGE THIS ON GPU!!
DOWNLOAD_MNIST = False

INPUT_DIM = 96                 # the dim of noise

DeviceID = [0]                 # this is decided by your gpu condition
G_LR = 0.001
D_LR = 0.001
IC_LR = 0.001
A_LR = 0.001

# SKIP_TURN = 1e9

def adjust_learning_rate(LR, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by * 0.1 every 10 epochs"""
    lr = LR * (0.1 ** (epoch // CHANGE_EPOCH))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def to_img(x):
    x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)

def bce_loss(input, target):
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def discriminator_loss(logits_real, logits_fake, device):    
    # Batch size.
    N = logits_real.size()
    
    # Target label vector, the discriminator should be aiming
    true_labels = torch.ones(N).to(device)
    # Discriminator loss has 2 parts: how well it classifies real images and how well it
    # classifies fake images.
    real_image_loss = bce_loss(logits_real, true_labels)
    fake_image_loss = bce_loss(logits_fake, 1 - true_labels)
    
    loss = real_image_loss + fake_image_loss
    
    return loss

def generator_loss(logits_fake, device):
    # Batch size.
    N = logits_fake.size()
    
    # Generator is trying to make the discriminator output 1 for all its images.
    # So we create a 'target' label vector of ones for computing generator loss.
    true_labels = torch.ones(N).to(device)
    
    # Compute the generator loss compraing 
    loss = bce_loss(logits_fake, true_labels)
    
    return loss

my_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def train(IC, D, G, A, IC_solver, D_solver, G_solver, A_solver, device, train_loader, num_epoch):
    # IC_gen = nn.Sequential(IC, nn.Tanh())
    iter_count = 0
    for epoch in range(num_epoch):
        if epoch % CHANGE_EPOCH == CHANGE_EPOCH - 1:
            adjust_learning_rate(IC_LR, IC_solver, epoch)
            adjust_learning_rate(A_LR, A_solver, epoch)
            adjust_learning_rate(G_LR, G_solver, epoch)
            adjust_learning_rate(D_LR, D_solver, epoch)

        for batch_idx, (data, target) in enumerate(train_loader):
            identity_images, identity = data.to(device), target.to(device)
            batch_size = identity_images.size(0) 
            
            if batch_idx % 2 == 0:
                attribute_images = identity_images
                r = 1
            else:
                r = 0.1

            # =============== train C ================
            IC_solver.zero_grad()

            IC.train()
            A.eval()
            G.eval()
            D.eval()

            # IC_feature will be reused later
            IC_id, IC_feature = IC(identity_images)
            LIC_loss = nn.CrossEntropyLoss()(IC_id, identity)
            LIC_loss.backward()
            IC_solver.step()
            IC_feature.detach_()
            # IC_feature = nn.Tanh()(IC_feature)

            # =============== train D ================
            D_solver.zero_grad()
            
            IC.eval()
            A.eval()
            G.eval()
            D.train()
            
            with torch.no_grad():
                # this IC will be resused later
                # IC_id = IC(identity_images).detach()
                A_output, _ = A(attribute_images)
                A_output.detach_()
            
            input_vector = torch.cat((IC_feature, A_output), 1)
            logits_real, _ = D(identity_images)

            with torch.no_grad():
                fake_images = G(input_vector).detach()
            logits_fake, _ = D(fake_images)

            LD_loss = discriminator_loss(logits_real, logits_fake, device)
            LD_loss.backward()        
            D_solver.step()

            # =============== train A ================
            A_solver.zero_grad()
            
            IC.eval()
            A.train()
            G.eval()
            D.eval()

            A_output, my_mean = A(attribute_images)
            input_vector = torch.cat((IC_feature, A_output), 1)
            fake_images = G(input_vector)
                       
            # a simplified version of LKL diver
            LKL_loss = 0.5 * (my_mean.pow(2)).sum()
            LGR_loss = 0.5 * ((fake_images - identity_images)**2).sum()
            LA_loss = LKL_loss + r * LGR_loss
            LA_loss.backward()
            A_solver.step()

            # =============== train G ================
            G_solver.zero_grad()
            
            IC.eval()
            A.eval()
            G.train()
            D.eval()

            with torch.no_grad():
                # this IC will be resused later
                # IC_id = IC_gen(identity_images).detach()
                A_output, _ = A(attribute_images)
                A_output.detach_()
                input_vector = torch.cat((IC_feature, A_output), 1)

            fake_images = G(input_vector)
            _, IC_fake_feature = IC(fake_images)
            _, fake_feature = D(fake_images)
            _, attribute_feature = D(attribute_images)

            LGR_loss = 0.5 * ((fake_images - identity_images)**2).sum()
            LGC_loss = 0.5 * ((IC_fake_feature - IC_feature)**2).sum()
            LGD_loss = 0.5 * ((attribute_feature - fake_feature)**2).sum()
            
            LG_loss = r * LGR_loss + LGC_loss + LGD_loss
            LG_loss.backward()
            
            # g_error = generator_loss(gen_logits_fake, device)
            # g_error.backward()
            G_solver.step()

            # =============== evaluation ================
            if (iter_count % SHOW_STEP == 0):
                print('Epoch: {}, Iter: {}, D: {:.4}, G:{:.4}, A:{:.4} '.format(epoch,iter_count,LD_loss.data.item(),LG_loss.data.item(), LA_loss.data.item()), end = '')
                print('LKL:{:.4}, LGR:{:.4}, LGC:{:.4}, LGD:{:.4}'.format(LKL_loss.data.item(), LGR_loss.data.item(), LGC_loss.data.item(), LGD_loss.data.item()) )

                IC.eval()
                A.eval()
                G.eval()
                D.eval()
                
                with torch.no_grad():
                  IC_id, IC_feature = IC(identity_images)
                  IC_feature.detach_()
                  # IC_feature = nn.Tanh()(IC_feature)

                  A_output, _ = A(attribute_images)
                  A_output.detach_()
                  input_vector = torch.cat((IC_feature, A_output), 1)
                  fake_images = G(input_vector)
                   
                save_dir = './gan_img/image_{}_{}'.format(epoch,iter_count)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                save_fake = to_img(fake_images.data.cpu().data)
                save_image(save_fake, os.path.join(save_dir,'fake.png'))
                save_atr = to_img(attribute_images.data.cpu().data)
                save_image(save_atr, os.path.join(save_dir,'atr.png'))
                save_id = to_img(identity_images.data.cpu().data)
                save_image(save_id, os.path.join(save_dir,'id.png'))

                # torch.save(D.state_dict(), './D.pkl')
                # torch.save(G.state_dict(), './G.pkl')
            iter_count += 1

def main():
    if not os.path.exists('./gan_img'):
        os.mkdir('./gan_img')

    train_data = torchvision.datasets.MNIST(
        root='./mnist',             # the location to save
        train=True,
        download=DOWNLOAD_MNIST,
        transform=my_transform
    )

    test_data = torchvision.datasets.MNIST(
        root='./mnist',
        train=False,
        download=DOWNLOAD_MNIST,
        transform=my_transform
    )

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)

    if USE_GPU:
        device = torch.device("cuda:" + str(DeviceID[0]))
    else:
        device = torch.device("cpu")

    from Classifier import Classifier
    IC = Classifier(num_feature=42, num_class=10)
    # IC.load_state_dict(torch.load('c_params.pkl'))
    if USE_GPU:
        IC = nn.DataParallel(IC, device_ids=DeviceID).to(device)

    from A_Encoder import A_Encoder
    A = A_Encoder(num_feature=54, use_gpu=USE_GPU)
    A.apply(initialize_weights)
    if USE_GPU:
        A = nn.DataParallel(A, device_ids=DeviceID).to(device)

    from Discriminator import Discriminator
    D = Discriminator()
    D.apply(initialize_weights)
    if USE_GPU:
        D = nn.DataParallel(D, device_ids=DeviceID).to(device)

    from Generator import Generator
    G = Generator(INPUT_DIM)
    G.apply(initialize_weights)
    if USE_GPU:
        G = nn.DataParallel(G, device_ids=DeviceID).to(device)
    
    A_solver = torch.optim.Adam(A.parameters(), lr=A_LR)
    # C_solver may not be used in this task
    IC_solver = torch.optim.Adam(IC.parameters(), lr=IC_LR)
    D_solver = torch.optim.Adam(D.parameters(), lr=D_LR, betas=(0.5, 0.999))
    G_solver = torch.optim.Adam(G.parameters(), lr=G_LR, betas=(0.5, 0.999))

    # model.load_state_dict(torch.load('params.pkl',  map_location=lambda storage, loc: storage))
    train(IC, D, G, A, IC_solver, D_solver, G_solver, A_solver, device, train_loader, EPOCH)
    # train(D, G, D_solver, G_solver, device, test_loader, epoch + 0.5)


if __name__ == '__main__':
    main()


