# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 17:40:14 2019

@author: mikap
"""
from wgangp import Discriminator 
from wgangp import Generator
from wgangp import get_data_loader
from wgangp import compute_gradient_penalty  


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
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.to(device)
    discriminator.to(device)

# Configure data loader
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5),
                         (.5, .5, .5))
])
    
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5,0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))

# ----------
#  Training
# ----------
n_epochs=50
n_critic=5
latent_dim=100
sample_interval=500
train, valid, test = get_data_loader("svhn", 256, image_transform)
batches_done = 0
for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(train):
        imgs = imgs.to(device)
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            
            if batches_done % sample_interval == 0:
                save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

            batches_done += n_critic

        if (i+1) % 20 ==0: # prints only every 20 bathces
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(train), d_loss.item(), g_loss.item())
            )

torch.save(discriminator.state_dict(),"./models/discriminator.pth")
torch.save(generator.state_dict(),"./models/generator.pth")