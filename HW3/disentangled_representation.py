import torch
from wgangp import Discriminator
from wgangp import Generator
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import save_image

# setup
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
np.random.seed(4)

# load models
Generator = Generator()
Generator.load_state_dict(torch.load("./models/generator.pth"))
Generator.to(device)
Generator.eval()

# Initial latent space
latent_dim = 100
batch_size = 1
z = Variable(Tensor(np.random.normal(0, 1, (batch_size, latent_dim)))).to(device)

im = Generator(z)
save_image(F.upsample(im,scale_factor=6),"./disentangled/GAN/original.png", normalize=True)

im = im[0].detach().cpu().numpy()

plt.imshow(im.transpose((1,2,0)))
plt.show()
####################
# noise
eps = 10
index = 10
for index in range(1000):
    noise = np.zeros(100)
    noise[index] = eps
    noise = Tensor(noise)

    im = Generator(z+noise)
    save_image(F.upsample(im,scale_factor=6),"./disentangled/GAN/image_%d.png" % index, normalize=True)

    # # Genera 1k images
    # np.random.seed(index)
    # z = Variable(Tensor(np.random.normal(0, 1, (batch_size, latent_dim)))).to(device)
    #
    # im = Generator(z)
    # save_image(im, "./fid/GAN/image_%d.png" % index, normalize=True)