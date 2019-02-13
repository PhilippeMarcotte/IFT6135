import numpy as np

import torch
from torch import cuda, nn
from torch.autograd import Variable


def validate(model, valid_loader):
    print("validation started")

    device = torch.device("cpu")
    if (cuda.is_available()):
        device = torch.device("cuda")

    model.to(device)
    model.eval()

    losses = []
    totalNumberItems = 0
    correctPredictions = 0

    with torch.no_grad():
        for (images, targets) in valid_loader:
            if (cuda.is_available()):
                images = images.to(device)
                targets = targets.to(device)

            images = Variable(images)
            targets = Variable(targets)

            output = model(images)

            predictedClasses = torch.argmax(output, dim=1)
            totalNumberItems += targets.size(0)
            correctPredictions += predictedClasses.eq(targets.data).cpu().sum().item()

            loss = nn.CrossEntropyLoss()
            loss = loss(output, targets).to(device)
            losses.append(loss.item())

    print("validation completed")

    return np.mean(losses), (correctPredictions / totalNumberItems)
