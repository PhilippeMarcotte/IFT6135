# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 17:40:14 2019

@author: mikap
"""
from wgangp import Discriminator 
from wgangp import Generator
from wgangp import get_data_loader
from wgangp import GradPenality


import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.autograd import Variable
import torch


# setup
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Loss weight for gradient penalty
lambda_gp = 15

# Initialize generator and discriminator
Generator = Generator()
Discriminator = Discriminator()

if cuda:
    Generator.to(device)
    Discriminator.to(device)

# Configure data loader
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5),
                         (.5, .5, .5))
])


# ----------
#  Training
# ----------
n_epochs=50
update_interval=4
latent_dim=100
sample_interval=400
train, valid, test = get_data_loader("svhn", 512, image_transform)
total_batches = 0
# Optimizers
optimizer_G = torch.optim.Adam(Generator.parameters(), lr=0.001, betas=(0.85,0.9))
optimizer_D = torch.optim.Adam(Discriminator.parameters(), lr=0.0007, betas=(0.85, 0.9))
for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(train):
        imgs = imgs.to(device)
        true_imgs = Variable(imgs.type(Tensor))

        # --- Train Discriminator ---
        optimizer_D.zero_grad()

        # Generate noise input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

        # Generate a batch of images
        fake_imgs = Generator(z)

        # Feed real and fake images to discriminator
        Dout_real = Discriminator(true_imgs)
        Dout_fake = Discriminator(fake_imgs)

        # Calculate GP
        GP = GradPenality(true_imgs.data, fake_imgs.data, Discriminator)

        # Calculate discriminant loss
        d_loss = -torch.mean(Dout_real) + torch.mean(Dout_fake) + lambda_gp * GP

        d_loss.backward()
        optimizer_D.step()
        optimizer_G.zero_grad()

        # Train the Generator every update_interval steps
        if i % update_interval == 0:

            # --- Train Generator ---

            # Calculate loss on generated images
            fake_imgs = Generator(z)
            Dout_fake = Discriminator(fake_imgs)
            g_loss = -torch.mean(Dout_fake)

            g_loss.backward()
            optimizer_G.step()

            
            if total_batches % sample_interval == 0:
                save_image(fake_imgs.data[:25], "images/%d.png" % total_batches, nrow=5, normalize=True)

            total_batches += update_interval

        if (i+1) % 20 ==0: # prints only every 20 bathces
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(train), d_loss.item(), g_loss.item())
            )

torch.save(Discriminator.state_dict(),"./models/discriminator.pth")
torch.save(Generator.state_dict(),"./models/generator.pth")