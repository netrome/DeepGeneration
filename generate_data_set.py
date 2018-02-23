import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import os

import settings

import utils.utils as u


out_dir = "./output/"
num_batches = 150

G = u.create_generator()
toRGB = nn.Conv2d(16, 2, 1)
G.load_state_dict(torch.load(os.path.join(settings.MODEL_PATH, "G.params")))
toRGB.load_state_dict(torch.load(os.path.join(settings.MODEL_PATH, "toRGB6.params")))
latent = Variable(torch.FloatTensor(settings.BATCH_SIZE, 128, 1, 1))

if settings.CUDA:
    toRGB.cuda()
    latent = latent.cuda()

num = 0
for i in range(num_batches):
    print("Generating batch {}/{}   ".format(i + 1, num_batches), end="\r")
    latent.data.normal_()
    batch = toRGB(G(latent))

    for rgb in batch:
        num += 1
        img = rgb[0].clamp(0, 1)
        map = rgb[1]

        torchvision.utils.save_image(img.data, os.path.join(out_dir, "image{}.png".format(num)))
        torchvision.utils.save_image(map.data, os.path.join(out_dir, "map{}.png".format(num)))




