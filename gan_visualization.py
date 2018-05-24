import torch
import torch.nn as nn

from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

import utils.utils as u
import utils.datasets as datasets

import settings
import os

image_grid = (6, 4)

num_images = image_grid[0] * image_grid[1]

G = u.create_generator()

toRGB = nn.Conv2d(16, 2, 1)
latent = Variable(torch.FloatTensor(num_images, 128, 1, 1))

torch.random.manual_seed(1337)

G.load_state_dict(torch.load(os.path.join(settings.MODEL_PATH, "G.params")))
toRGB.load_state_dict(torch.load(os.path.join(settings.MODEL_PATH, "toRGB6.params")))

if settings.CUDA:
    toRGB.cuda()
    latent = latent.cuda()

latent.data.normal_()
fake = toRGB(G(latent))

single = make_grid(fake[:, 0].data.cpu().contiguous().view(num_images, 1, 256, 256), nrow=image_grid[0]) 
save_image(single, "/tmp/fake.png")

print("Done")
