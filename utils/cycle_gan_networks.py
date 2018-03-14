""" Networks inspired by the cycleGAN paper """
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniBatchSTD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        shape = list(img.shape)
        shape[1] = 1
        minibatch_std = img.std(0).mean().expand(shape) + 1e-6
        out = torch.cat([img, minibatch_std], dim=1)
        return out



class PatchGANDiscriminator(nn.Module):
    # Don't use padding in discriminator, keep it simple
    def __init__(self):
        super().__init__()

        seq = [
                nn.Conv2d(16, 64, 4, stride=2), 
                nn.Conv2d(64, 64, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 128, 4, stride=2),
                nn.Conv2d(128, 128, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(128, 256, 4, stride=2),
                nn.Conv2d(256, 256, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(256, 512, 4, stride=2),
                MiniBatchSTD(),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(513, 512, 4, stride=2),
                nn.Conv2d(512, 1, 1),
                ]


        self.model = nn.Sequential(*seq)

    def forward(self, batch):
        return self.model(batch).mean(2).mean(2)  # Average result over patches


class WannaBeCycleGANGenerator(nn.Module):
    pass  # Todo: sequential simple upsampling model


