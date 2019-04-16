import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, n_input = 2, num_classes=1):
        super(Discriminator,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_input,100),
            nn.ReLU(True),
            nn.Linear(100,100),
            nn.ReLU(True),
            nn.Linear(100,num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

