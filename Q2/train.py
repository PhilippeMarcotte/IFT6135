import numpy as np

import torch
import torch.cuda as cuda
import torch.nn as nn
from torch.autograd import Variable


def train(model, train_loader, optimizer):
    print("training started")

    device = torch.device("cpu")
    if (cuda.is_available()):
        device = torch.device("cuda")

    model.to(device)
    model.train()

    losses = []

    totalNumberItems = 0
    correctPredictions = 0

    for (images, targets) in train_loader:
        if (cuda.is_available()):
            images = images.to(device)
            targets = targets.to(device)

        images = Variable(images)
        targets = Variable(targets)

        optimizer.zero_grad()

        output = model(images)

        loss = nn.CrossEntropyLoss()
        loss = loss(output, targets).to(device)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        predictedClasses = torch.argmax(output, 1)
        totalNumberItems += targets.size(0)
        correctPredictions += predictedClasses.eq(targets.data).cpu().sum().item()

    print("training completed")

    return np.mean(losses), (correctPredictions / totalNumberItems)
