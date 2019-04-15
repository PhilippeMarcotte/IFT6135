import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, num_classes=1):
        super(Discriminator,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2,100),
            nn.ReLU(True),
            nn.Linear(100,100),
            nn.ReLU(True),
            nn.Linear(100,num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def GradPenality(p, q, Discriminator):
    # random varibale for interpolation
    a = Tensor(np.random.random((p.shape)))
    #interpolation
    z = (a*p + (1-a)*q).requires_grad(True)
    D_z = Discriminator(z)
    # Calculate gradient
    grad_z = torch.autograd.gard(
        outputs = D_z,
        inputs = z,
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )
    grad_z = grad_z.view(grad_z.shape[0],-1)
    penality = ((grad_z.norm(2, dim=1)-1)**2).mean()
    return penality
