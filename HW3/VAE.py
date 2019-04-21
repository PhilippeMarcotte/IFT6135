import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, cuda
from torch.optim import Adam

from Utils import save_model
from mnist_loader import get_data_loader
from tqdm import tqdm

class Squeeze(nn.Module):
    def __init__(self, *args):
        super(Squeeze, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.squeeze()

class Unsqueeze(nn.Module):
    def __init__(self, *args):
        super(Unsqueeze, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1, 256, 1, 1)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 256, kernel_size=(5, 5)),
            nn.ELU(),
            Squeeze(),
            nn.Linear(in_features=256, out_features=100*2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=100, out_features=256),
            Unsqueeze(),
            nn.ELU(),
            nn.Conv2d(256, 64, kernel_size=(5, 5), padding=(4,4)),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=(3,3), padding=(2,2)),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=(3,3), padding=(2,2)),
            nn.ELU(),
            nn.Conv2d(16, 1, kernel_size=(3,3), padding=(2,2)),
            nn.Sigmoid()
        )

    def forward(self, x):
        log_sigma, mu, z = self.sample_latent_from(x)
        return self.decoder(z), mu, log_sigma

    def sample_latent_from(self, x):
        q_params = self.encoder(x)
        mu = q_params[:, 100:]
        log_sigma = q_params[:, :100]
        z = self.reparameterize(mu, log_sigma)
        return log_sigma, mu, z

    def reparameterize(self, mu, log_sigma):
        sigma = torch.exp(log_sigma) + math.exp(-7)

        e = torch.randn_like(sigma)
        return mu + sigma * e

    def importance_sampling(self, x, k):
        Pis = []
        for i in range(k):
            log_sigma, mu, z = self.sample_latent_from(x)
            x_ = self.decoder(z)
            Pis.append(self.getPx_z(x_, x) + self.getPz(z) - self.getQz_x(z, mu, torch.exp(log_sigma)))

        Pis = torch.stack(Pis).transpose(0, 1)
        max,_ = torch.max(Pis, 1)
        return max + torch.logsumexp(Pis - max.unsqueeze(1), 1) - np.log(k)

    def getPz(self, z):
        covar = torch.eye(100, 100).to(z.device)
        mu = torch.zeros(100).to(z.device)
        return torch.distributions.multivariate_normal.MultivariateNormal(mu, covar).log_prob(z)

    def getQz_x(self, z, mu, sigma):
        covar = torch.diag_embed(sigma.unsqueeze(1)).squeeze()
        return torch.distributions.multivariate_normal.MultivariateNormal(mu, covar).log_prob(z)

    def getPx_z(self, x_, x):
        BCE = nn.BCELoss(reduction="none")
        return -BCE(x_.view(-1, 784), x.view(-1, 784)).sum(1)

def ELBOWLoss(x, x_, mu, log_sigma):
    KL = KLDivergence(mu, log_sigma)

    BCE = nn.BCELoss(reduction="sum")
    logpx_z = BCE(x_, x)
    return (logpx_z + KL) #batch mean loss computed in BCE

def KLDivergence(mu, log_sigma):
    return -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())

def train(VAE, train_loader, optimizer, device):
    VAE.train()
    losses = []
    mean_loss = 0
    total = 0
    total_step = len(train_loader)
    for batch_id, batchX in enumerate(train_loader):
        batchX = batchX.to(device)
        optimizer.zero_grad()
        batchX_, mu, log_sigma = VAE(batchX)
        loss = ELBOWLoss(batchX.view(-1,784), batchX_.view(-1,784), mu, log_sigma)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        mean_loss += loss.item()
        total += batchX.shape[0]

        """if (batch_id + 1) % 100 == 0:
            print('Step [{}/{}], Loss: {:.4f}({:.4f})'
                  .format(batch_id + 1, total_step, loss.item()/batchX.shape[0], mean_loss / total))"""

    mean_loss = mean_loss / total
    return mean_loss

def validate(model, validation_loader, device):
    model.eval()
    mean_loss = 0
    total = 0
    with torch.no_grad():
        for batch_id, batchX in enumerate(validation_loader):
            batchX = batchX.to(device)
            batchX_, mu, log_sigma = model(batchX)
            loss = ELBOWLoss(batchX.view(-1,784), batchX_.view(-1,784), mu, log_sigma)
            mean_loss += loss.item()
            total += batchX.shape[0]

        mean_loss = mean_loss / total
        #print('Validation loss: {:.4f}'.format(mean_loss))
    return mean_loss

def importance_sampling(model, data_loader, device):
    model.eval()

    with torch.no_grad():
        importance_samples = []
        for batch_id, batchX in enumerate(data_loader):
            batchX = batchX.to(device)
            approximate_per_batch.append(torch.mean(model.importance_sampling(batchX, 200)))

        return torch.mean(torch.stack(approximate_per_batch))

if __name__ == "__main__":

    device = torch.device("cpu")
    if cuda.is_available():
        device = torch.device("cuda")

    train_loader, valid_loader, test_loader = get_data_loader("binarized_mnist", 64)

    model = VAE()
    model.to(device)
    optimizer = Adam(params=model.parameters(), lr=3 * 10 ** (-4))
    num_epochs = 20
    trainLosses = []
    validLosses = []
    caca = importance_sampling(model, test_loader, device)
    for epoch in range(num_epochs):
        print("-------------- Epoch # " + str(epoch+1) + " --------------")

        trainLoss = train(model, train_loader, optimizer, device)
        trainLosses.append(trainLoss)
        print("Epoch train loss: {:.4f}".format(trainLoss))

        validationLoss = validate(model, valid_loader, device)
        validLosses.append(validationLoss)
        print("Epoch validation loss: {:.4f}".format(validationLoss))

        save_model(model, optimizer, epoch, trainLoss, validationLoss)

    imp_sampling_mean = importance_sampling(model, test_loader, device)
    print(imp_sampling_mean)

    plt.plot(np.arange(num_epochs), trainLosses)
    plt.plot(np.arange(num_epochs), validLosses)
    plt.legend(["Training","Validation"])
    plt.ylabel("ELBO Loss")
    plt.xlabel("Epoch number")
    plt.savefig("./results/VAE_training_20_epochs.png")
    plt.show()