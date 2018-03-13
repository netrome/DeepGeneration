""" Networks inspired by the cycleGAN paper """
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchGANDiscriminator(nn.Module):
    # Don't use padding in discriminator, keep it simple
    def __init__(self):
        super().__init__()

        seq = [
                nn.Conv2d(16, 64, 4, stride=2), 
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 128, 4, stride=2),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(128, 256, 4, stride=2),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(256, 512, 4, stride=2),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(512, 1, 4, stride=2)
                ]


        self.model = nn.Sequential(*seq)

    def forward(self, batch):
        return self.model(batch).mean(2).mean(2)  # Average result over patches


class WannaBeCycleGANGenerator(nn.Module):
    pass  # Todo: sequential simple upsampling model


