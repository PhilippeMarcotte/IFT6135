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
z1 = Variable(Tensor(np.random.normal(0, 1, (batch_size, latent_dim)))).to(device)

###################################
np.random.seed(83) #2
###################################

z2 = Variable(Tensor(np.random.normal(0, 1, (batch_size, latent_dim)))).to(device)
im1 = Generator(z1)
im2 = Generator(z2)
# save_image(F.upsample(im1,scale_factor=6),"./interpolations/z1.png", normalize=True)
# save_image(F.upsample(im2,scale_factor=6),"./interpolations/z2.png", normalize=True)
####################
# interpolations

for i in range(11):
    alpha = 0.1 * i
    # z interpol
    z_ = alpha*z1 + (1-alpha)*z2

    im = Generator(z_)
    save_image(F.upsample(im,scale_factor=6),"./interpolations/VAE/z interpol/image_%d.png" % i, normalize=True)

    # x interpol
    x_ = alpha * im1 + (1 - alpha) * im2

    save_image(F.upsample(x_, scale_factor=6), "./interpolations/VAE/x interpol/image_%d.png" % i, normalize=True)

im1 = im1[0].detach().cpu().numpy()
plt.imshow(im1.transpose((1,2,0)))
plt.show()
im2 = im2[0].detach().cpu().numpy()
plt.imshow(im2.transpose((1,2,0)))
plt.show()