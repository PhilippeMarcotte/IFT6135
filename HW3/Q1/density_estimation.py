#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""


from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import samplers
import discriminators
import distances
import torch.nn.functional as F
from Q1.Q1_training import training_loop
import Losses


# plot p0 and p1
plt.figure()

# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
plt.hist(f(xx), 100, alpha=0.5, density=1)
plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5,5)
# exact
xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.plot(xx, N(xx))
plt.show()

############### import the sampler ``samplers.distribution4'' 
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######

loss, D = training_loop(Losses.BCELoss2(), 0, distribution=4,learning_rate=0.01, num_epochs=15000)
torch.save(D,"model.pt")

# If using saved model
# D = torch.load("./model.pt")
# D.eval()

############### plotting things
############### (1) plot the output of your trained discriminator 
############### (2) plot the estimated density contrasted with the true density


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
input = (torch.tensor(xx).float().to(device)).view(1000,-1)
r = torch.sigmoid(D(input)).cpu().detach().numpy() # evaluate xx using your discriminator; replace xx with the output
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(xx,r)
plt.title(r'$D(x)$')


estimate = np.multiply(np.transpose(np.asmatrix(N(xx))),(r/(1-r)))#np.ones_like(xx)*0.2 # estimate the density of distribution4 (on xx) using the discriminator;
                                # replace "np.ones_like(xx)*0." with your estimate

plt.subplot(1,2,2)
plt.plot(xx,estimate,'ro')
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.legend(['Estimated','True'])
plt.title('Estimated vs True')
plt.savefig("Q1.4.png")
plt.show()










