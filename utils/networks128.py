import torch.nn as nn
import torch.nn.functional as F
import torch

from .progressive_networks import TrivialDownBlock, DownsamplingDownBlock, TrivialUpBlock, UpsamplingUpBlock
from .cycle_gan_networks import MiniBatchSTD

# Encoders ---------------------------------------
class Encoding_layer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc = nn.Linear(size, 128)
    def forward(self, batch):
        return self.fc(batch), None

class Debug_Layer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, batch):
        print(batch.shape)
        return batch

class Flatten(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
    def forward(self, batch):
        return batch.view(-1, self.size)

Encoder = nn.Sequential(
        TrivialDownBlock(16, 32, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        TrivialDownBlock(32, 64, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        TrivialDownBlock(64, 128, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        TrivialDownBlock(128, 256, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        TrivialDownBlock(256, 512, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        MiniBatchSTD(),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(513, 512, 3, padding=1),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(512, 512, 4),
        nn.LeakyReLU(negative_slope=0.2),
        Flatten(512),
        Encoding_layer(512)
        # A sigmoid could be nice here
        )

# Decoders/generators ----------------------------

Generator = nn.Sequential(
        nn.ConvTranspose2d(128, 512, 4),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(512, 512, 3, padding=1),
        nn.LeakyReLU(negative_slope=0.2),
        TrivialUpBlock(512, 512, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        TrivialUpBlock(512, 256, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        TrivialUpBlock(256, 128, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        TrivialUpBlock(128, 64, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        TrivialUpBlock(64, 32, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2)
        )

# Discriminator
Discriminator = nn.Sequential(
        TrivialDownBlock(32, 64, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        TrivialDownBlock(64, 128, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        TrivialDownBlock(128, 256, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        TrivialDownBlock(256, 512, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        TrivialDownBlock(512, 512, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        MiniBatchSTD(),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(513, 512, 3, padding=1),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(512, 512, 4),
        nn.LeakyReLU(negative_slope=0.2),
        Flatten(512),
        nn.Linear(512, 1),
        )

# Image to image models --------------------------------------

