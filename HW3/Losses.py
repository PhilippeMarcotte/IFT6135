import numpy as np
import torch
import torch.nn as nn
import samplers
import discriminators
import torch.nn.functional as F
from torch.autograd import Variable

class JSDLoss(nn.Module):
    def __init__(self):
        super(JSDLoss,self).__init__()
        self.criterion = torch.nn.BCELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def forward(self, outputs_real,outputs_fake, labels_real, labels_fake, p,q, D):
        return (torch.log(torch.tensor([2.0])).to(self.device) + 0.5*self.criterion(outputs_real, labels_real) + 0.5*self.criterion(outputs_fake, labels_fake))


class WassersteinLoss(nn.Module):
    def __init__(self, lambda_weight):
        super(WassersteinLoss,self).__init__()
        self.lamda_weight = lambda_weight

    def GradPenality(self, p, q, Discriminator):
        # random varibale for interpolation
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        a = torch.FloatTensor(np.random.random((p.shape))).to(device)
        # interpolation
        z = (a * p + (1 - a) * q).requires_grad_(True)
        D_z = Discriminator(z)
        fake = Variable(torch.FloatTensor(p.shape[0], 1).fill_(1.0).to(device), requires_grad=False)
        # Calculate gradient
        grad_z = torch.autograd.grad(
            outputs=D_z,
            inputs=z,
            grad_outputs= fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0] # extract out of tuple
        grad_z = grad_z.view(grad_z.shape[0], -1)
        penality = ((grad_z.norm(2, dim=1) - 1) ** 2).mean()
        return penality
    def forward(self, outputs_real,outputs_fake, labels_real, labels_fake, p,q, D):
        GP = self.GradPenality(p,q,D)
        return -torch.mean(outputs_real) + torch.mean(outputs_fake) + self.lamda_weight*GP