# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 17:02:12 2019

@author: mikap
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from Utils import save_model
import os
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.autograd import Variable
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.datasets
from torch.utils.data import dataset
# from torch.nn.modules import upsampling
# from torch.functional import F

#Setup
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
os.makedirs("images_VAE", exist_ok=True)
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

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5, 5)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=(5, 5)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 256, kernel_size=(5, 5)),
            nn.ELU(),
            Squeeze(),
            nn.Linear(in_features=256, out_features=100*2)
        )

        self.decoder = nn.Sequential(
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

    def forward(self, x):
        q_params = self.encoder(x)
        mu = q_params[:, 100:]
        log_sigma = q_params[:, :100]
        z = self.reparameterize(mu, log_sigma)
        return self.decoder(z), mu, log_sigma

    def reparameterize(self, mu, log_sigma):
        sigma = torch.exp(0.5*log_sigma) + math.exp(-7)

        e = torch.randn_like(sigma)
        return mu + sigma * e


def ELBOWLoss(x, x_, mu, log_sigma):
    KL = KLDivergence(mu, log_sigma)

    MSE = nn.MSELoss(reduction="sum")
    logpx_z = MSE(x_, x)
    return (logpx_z + KL) #batch mean loss computed in BCE

def KLDivergence(mu, log_sigma):
    return -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())


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

def train(VAE, train_loader, optimizer, device):
    VAE.train()
    losses = []
    mean_loss = 0
    total = 0
    total_step = len(train_loader)
    for batch_id, (batchX, _) in enumerate(train_loader):

        batchX = batchX.to(device)
        optimizer.zero_grad()
        batchX_, mu, log_sigma = VAE(batchX)
        loss = ELBOWLoss(batchX.view(-1,3,1024), batchX_.view(-1,3,1024), mu, log_sigma)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        mean_loss += loss.item()
        total += batchX.shape[0]

        if (batch_id + 1) % 100 == 0:
            print('Step [{}/{}], Loss: {:.4f}({:.4f})'
                  .format(batch_id + 1, total_step, loss.item()/batchX.shape[0], mean_loss / total))

    mean_loss = mean_loss / total
    return mean_loss


def validate(VAE, validation_loader, device):
    VAE.eval()
    mean_loss = 0
    total = 0
    with torch.no_grad():
        for batch_id, (batchX, _) in enumerate(validation_loader):
            batchX = batchX.to(device)
            batchX_, mu, log_sigma = VAE(batchX)
            loss = ELBOWLoss(batchX.view(-1,3,1024), batchX_.view(-1,3,1024), mu, log_sigma)
            mean_loss += loss.item()
            total += batchX.shape[0]

        mean_loss = mean_loss / total
        print('Validation loss: {:.4f}'.format(mean_loss))
    return mean_loss
