import numpy as np
import torch
import torch.nn as nn
import samplers
import discriminators
import torch.nn.functional as F

def training_loop(LossFct, x, distribution=1, learning_rate = 0.0001, num_epochs = 50000):
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Distributions properties
    if distribution == 1:
        p_gen = samplers.distribution1(0)
        q_gen = samplers.distribution1(x)
        nb_input = 2
    elif distribution == 4:
        q_gen = samplers.distribution3(2048)
        p_gen = samplers.distribution4(2048)
        nb_input = 1
    p_gen.send(None)
    q_gen.send(None)


    D = discriminators.Discriminator(n_input=nb_input).to(device)

    # Loss and optimizer
    optimizer = torch.optim.SGD(D.parameters(), lr=learning_rate)

    # Train the model
    trainLoss = []
    trainAcc = []
    meanLoss = 0
    correct = 0
    total = 0
    log_frequency = 100
    for epoch in range(num_epochs):
        #     exp_lr_scheduler.step()
        p = torch.from_numpy(p_gen.send(0)).float().to(device)
        q = torch.from_numpy(q_gen.send(x)).float().to(device)
        labels_real = torch.ones(p.shape[0]).to(device)
        labels_fake = torch.zeros(q.shape[0]).to(device)
        # Forward pass
        outputs_real = torch.sigmoid(D(p))
        outputs_fake = torch.sigmoid(D(q))

        predicted_real = (outputs_real.data > 0.5).float().squeeze()
        predicted_fake = (outputs_fake.data > 0.5).float().squeeze()
        total += 2*labels_real.size(0)
        correct_this_batch = (predicted_real == labels_real).sum().item() + (predicted_fake == labels_fake).sum().item()
        correct += correct_this_batch
        loss = LossFct.forward(outputs_real,outputs_fake, labels_real, labels_fake, p,q, D)#(torch.log(torch.tensor([2.0])).to(device) + 0.5*criterion(outputs_real, labels_real) + 0.5*criterion(outputs_fake, labels_fake))
        meanLoss += loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % log_frequency == 0:
            print('Epoch [{}/{}]'.format(epoch, num_epochs))
            print('Loss: {:.4f}({:.4f}), Acc: {:.3f}({:.3f})'
                  .format(loss.item(), meanLoss / (epoch + 1), correct_this_batch * 100 / (2*labels_real.size(0)),
                          correct * 100 / total))
        trainLoss.append(meanLoss / (epoch + 1))
        trainAcc.append(100 * correct / total)
    return loss, D