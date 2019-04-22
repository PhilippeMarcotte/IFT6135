# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:53:56 2019

@author: mikap
"""
import os
import numpy as np

from torch.autograd import Variable
from torch import nn

import torch
import torchvision.datasets
from torch.utils.data import dataset



# setup
os.makedirs("images", exist_ok=True)

img_shape = (3,32,32)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class Squeeze(nn.Module):
    def __init__(self, *args):
        super(Squeeze, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.squeeze()

class Unsqueeze(nn.Module):
    def __init__(self, *args):
        super(Unsqueeze, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1, 256, 1, 1)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.gen=nn.Sequential(
            nn.Linear(in_features=100, out_features=256),
            Unsqueeze(),
            nn.ELU(),
            nn.Conv2d(256, 64, kernel_size=(5, 5), padding=(4,4)),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=(3,3), padding=(2,2)),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=(3,3), padding=(3,3)),
            nn.ELU(),
            nn.Conv2d(16, 3, kernel_size=(3,3), padding=(3,3)),
            nn.Tanh()
                
                
                )

    def forward(self, z):
        img = self.gen(z)
        img = img.view(img.shape[0], 3, 32,32)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.dis = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5, 5)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=(5, 5)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 256, kernel_size=(5, 5)),
            nn.ELU(),
            Squeeze(),
            nn.Linear(in_features=256, out_features=1)
                
        )

    def forward(self, img):
        #img_flat = img.view(img.shape[0], -1)
        validity = self.dis(img)
        return validity



def get_data_loader(dataset_location, batch_size, image_transform):
    trainvalid = torchvision.datasets.SVHN(
        dataset_location, split='train',
        download=True,
        transform=image_transform
    )

    trainset_size = int(len(trainvalid) * 0.9)
    trainset, validset = dataset.random_split(
        trainvalid,
        [trainset_size, len(trainvalid) - trainset_size]
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
    )

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(
            dataset_location, split='test',
            download=True,
            transform=image_transform
        ),
        batch_size=batch_size,
    )

    return trainloader, validloader, testloader


def GradPenality( p, q, Discriminator):
    # random varibale for interpolation
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    a = torch.FloatTensor(np.random.random((p.shape))).to(device)
    # interpolation
    z = (a * p + (1 - a) * q).requires_grad_(True)
    D_z = Discriminator(z)
    fake = Variable(torch.FloatTensor(p.shape[0], 1).fill_(1.0).to(device), requires_grad=False)
    # Calculate gradient
    grad_z = torch.autograd.grad(
        outputs=D_z,
        inputs=z,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # extract out of tuple
    grad_z = grad_z.view(grad_z.shape[0], -1)
    return ((grad_z.norm(2, dim=1) - 1) ** 2).mean()



