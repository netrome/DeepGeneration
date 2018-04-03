import torch
import torch.nn as nn

from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

import utils.utils as u
import utils.datasets as datasets

import settings
import os

G = u.create_generator()

toRGB = nn.Conv2d(16, 2, 1)
latent = Variable(torch.FloatTensor(8, 128, 1, 1))

torch.random.manual_seed(1337)

G.load_state_dict(torch.load(os.path.join(settings.MODEL_PATH, "G.params")))
toRGB.load_state_dict(torch.load(os.path.join(settings.MODEL_PATH, "toRGB6.params")))

if settings.CUDA:
    toRGB.cuda()
    latent = latent.cuda()

latent.data.normal_()
fake = toRGB(G(latent))

single = make_grid(fake[:, 0].data.cpu().contiguous().view(8, 1, 256, 256))
save_image(single, "/tmp/fake.png")

print("Done")
