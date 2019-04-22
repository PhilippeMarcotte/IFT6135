import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, n_input = 2, num_classes=1):
        super(Discriminator,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_input,2000),
            nn.ReLU(True),
            nn.Linear(2000,500),
            nn.ReLU(True),
            nn.Linear(500,num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

