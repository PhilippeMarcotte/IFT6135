from HW3.Q2.VAE import importance_sampling, VAE
from HW3.Q2.mnist_loader import get_data_loader
import torch
import numpy as np

checkpoint = None#torch.load("checkpoint.pth")
model = VAE()

np.random.seed(0)
torch.cuda.manual_seed_all(0)
torch.manual_seed(0)
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

model = model.to(device)
train_loader, valid_loader, test_loader = get_data_loader("binarized_mnist", 64)

valid_lle = importance_sampling(model, valid_loader, device)
print(valid_lle)

test_lle = importance_sampling(model, test_loader, device)
print(test_lle)
