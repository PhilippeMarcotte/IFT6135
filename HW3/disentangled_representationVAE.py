import torch
from VAEQ3 import VAE
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import save_image

# setup
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
np.random.seed(21)

# load models
Generator = VAE()
Generator.load_state_dict(torch.load("./models/decoderVAE.pth"))
Generator.to(device)
Generator.eval()
Generator = Generator.decoder

# Initial latent space
latent_dim = 100
batch_size = 1
z = Variable(Tensor(np.random.normal(0, 1, (batch_size, latent_dim)))).to(device)

im = Generator(z)
save_image(F.upsample(im,scale_factor=6),"./disentangled/VAE/original.png", normalize=True)

im = im[0].detach().cpu().numpy()

plt.imshow(im.transpose((1,2,0)))
plt.show()
####################
# noise
eps = 5
index = 10
for index in range(100):
    noise = np.zeros(100)
    noise[index] = eps
    noise = Tensor(noise)

    im = Generator(z + noise)
    save_image(F.upsample(im, scale_factor=6), "./disentangled/VAE/image_%d.png" % index, normalize=True)

    # np.random.seed(index)
    # z = Variable(Tensor(np.random.normal(0, 1, (batch_size, latent_dim)))).to(device)
    # im = Generator(z)
    # save_image(im, "./fid/VAE/fake/image_%d.png" % index)