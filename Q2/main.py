import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torchvision import datasets
from torchvision.transforms import transforms

from Utils import save_model
from model import CNN
from train import train
from validate import validate

dataset_transformation = transforms.ToTensor()
trainSet = datasets.MNIST(root='./data', train=True, download=True, transform=dataset_transformation)
testSet = datasets.MNIST(root='./data', train=False, download=True, transform=dataset_transformation)

train_loader = torch.utils.data.DataLoader(trainSet, batch_size=64, shuffle=True, num_workers=2)
valid_loader = torch.utils.data.DataLoader(testSet, batch_size=64, shuffle=True, num_workers=2)

trainingLosses = []
trainingAccuracies = []
validationLosses = []
validationAccuracies = []

model = CNN()
optimizer = optim.SGD(model.parameters(), lr=0.1)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total number of trainable parameters: ", total_params)

for i in range(10):
    print("\n\n--------- staring iteration: ", i)

    trainLoss, trainAcc = train(model, train_loader, optimizer)
    validLoss, validAcc = validate(model, valid_loader)

    trainingLosses.append(trainLoss)
    trainingAccuracies.append(trainAcc)

    validationLosses.append(validLoss)
    validationAccuracies.append(validAcc)

    print("training loss", trainLoss)
    print("training accuracy", trainAcc)

    print("\nvalidation loss", validLoss)
    print("validation accuracy", validAcc)

    print("\nsaving checkpoint ")
    save_model(model, optimizer, i, trainingLosses, trainingAccuracies, validationLosses, validationAccuracies)

"""checkpoint = load_model()

trainingLosses = checkpoint["trainingLosses"]
trainingAccuracies = checkpoint["trainingAccuracies"]
validationLosses = checkpoint["validationLosses"]
validationAccuracies = checkpoint["validationAccuracies"]"""

x = np.linspace(0, 10, 10)

plt.subplot(2, 2, 1)
plt.plot(x, trainingLosses)
plt.ylabel('Train Loss')
plt.xlabel('Epochs')

plt.subplot(2, 2, 2)
plt.plot(x, trainingAccuracies)
plt.ylabel('Train Accuracy')
plt.xlabel('Epochs')

plt.subplot(2, 2, 3)
plt.plot(x, validationLosses)
plt.ylabel('Validation Loss')
plt.xlabel('Epochs')

plt.subplot(2, 2, 4)
plt.plot(x, validationAccuracies)
plt.ylabel('Validation Accuracy')
plt.xlabel('Epochs')

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(12, 12)
matplotlib.pyplot.show()
