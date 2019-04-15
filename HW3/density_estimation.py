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
from distances import distance
import torch.nn.functional as F



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

############### Q1.3 ########################
JSD = []
WSD = []
x = np.linspace(-1,1,21)

p_gen = samplers.distribution1(0)
# q_gen = samplers.distribution1(0)
p_gen.send(None)
#q_gen.send(None)
for i in range(21):
    p = p_gen.send(0)
    q_gen = samplers.distribution1(x[i])
    q_gen.send(None)
    q = q_gen.send(x[i])
    # distances
    JSD.append(distance.jsd(p,q))
    WSD.append(distance.wasserstein(p,q))

plt.plot(x,JSD)
plt.plot(x,WSD)
plt.show()
############### import the sampler ``samplers.distribution4'' 
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

learning_rate = 0.001
D = discriminators.Discriminator().to(device)

# Loss and optimizer
criterion = torch.nn.functional.binary_cross_entropy
optimizer = torch.optim.SGD(D.parameters(), lr=learning_rate)

# Train the model
p_gen = samplers.distribution1(0)
q_gen = samplers.distribution1(-1)
p_gen.send(None)
q_gen.send(None)

trainLoss = []
validLoss = []
validAcc = []
total_step = 2
trainAcc = []
best_acc = 0
num_epochs = 100
for epoch in range(num_epochs):
    #     exp_lr_scheduler.step()
    print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
    meanLoss = 0
    correct = 0
    total = 0
    for i in range(2):
        p = torch.from_numpy(p_gen.send(0)).float().to(device)
        q = torch.from_numpy(q_gen.send(-1)).float().to(device)
        labels_real = torch.ones(p.shape[0]).to(device)
        labels_fake = torch.zeros(q.shape[0]).to(device)
        # Forward pass
        outputs_real = D(p)
        outputs_fake = D(q)

        _, predicted_real = torch.max(outputs_real.data, 1)
        _, predicted_fake = torch.max(outputs_fake.data, 1)
        total += 2*labels_real.size(0)
        correct_this_batch = (predicted_real.float() == labels_real).sum().item() + (predicted_fake.float() == labels_fake).sum().item()
        correct += correct_this_batch
        loss = (torch.log(torch.tensor([2.0])).to(device) + 0.5*criterion(F.sigmoid(outputs_real), labels_real) + 0.5*criterion(F.sigmoid(outputs_fake), labels_fake))
        meanLoss += loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 1 == 0:
            print('Step [{}/{}], Loss: {:.4f}({:.4f}), Acc: {:.3f}({:.3f})'
                  .format(i + 1, total_step, loss.item(), meanLoss / (i + 1), correct_this_batch * 100 / (2*labels_real.size(0)),
                          correct * 100 / total, ))
        trainLoss.append(meanLoss / (i + 1))
        trainAcc.append(100 * correct / total)
















############### plotting things
############### (1) plot the output of your trained discriminator 
############### (2) plot the estimated density contrasted with the true density



# r = xx # evaluate xx using your discriminator; replace xx with the output
# plt.figure(figsize=(8,4))
# plt.subplot(1,2,1)
# plt.plot(xx,r)
# plt.title(r'$D(x)$')
#
# estimate = np.ones_like(xx)*0.2 # estimate the density of distribution4 (on xx) using the discriminator;
#                                 # replace "np.ones_like(xx)*0." with your estimate
# plt.subplot(1,2,2)
# plt.plot(xx,estimate)
# plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
# plt.legend(['Estimated','True'])
# plt.title('Estimated vs True')











