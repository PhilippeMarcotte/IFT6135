# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 17:40:14 2019

@author: mikap
"""

from VAEQ3 import VAE
from VAEQ3 import train
from VAEQ3 import validate
from VAEQ3 import get_data_loader
from Utils import save_model

import numpy as np
import matplotlib.pyplot as plt


from torchvision.utils import save_image
from torch.autograd import Variable
from torch.optim import Adam
import torchvision.transforms as transforms
import torch



# ----------
#  Training
# ----------

# setup
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Configure data loader
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5),
                         (.5, .5, .5))
])
train_loader, valid_loader, test_loader = get_data_loader("svhn", 256, image_transform)

VAE = VAE()
VAE.to(device)
optimizer = Adam(params=VAE.parameters(), lr=0.0001)
num_epochs = 50
trainLosses = []
validLosses = []
for epoch in range(num_epochs):
    print("-------------- Epoch # " + str(epoch+1) + " --------------")

    trainLoss = train(VAE, train_loader, optimizer, device)
    trainLosses.append(trainLoss)
    print("Epoch train loss: {:.4f}".format(trainLoss))

    validationLoss = validate(VAE, valid_loader, device)
    validLosses.append(validationLoss)
    decoder_fake=VAE.decoder
    z = Variable(Tensor(np.random.normal(np.zeros(100),np.ones(100) , (25, 100))))
    fake_imgs=decoder_fake(z)
    save_image(fake_imgs.data, "images_VAE/%d.png" % epoch,nrow=5, normalize=True)

save_model(VAE, optimizer, epoch, trainLoss, validationLoss)
torch.save(VAE, "decoderVAE.pth")

plt.plot(np.arange(num_epochs), trainLosses)
plt.plot(np.arange(num_epochs), validLosses)
plt.legend(["Training","Validation"])
plt.ylabel("ELBO Loss")
plt.xlabel("Epoch number")
plt.savefig("./results/VAE_training_20_epochs.png")
plt.show()