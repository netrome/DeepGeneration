import torch.nn as nn
import torch.nn.functional as F
import torch

from .progressive_networks import TrivialDownBlock, DownsamplingDownBlock, TrivialUpBlock, UpsamplingUpBlock
from .cycle_gan_networks import MiniBatchSTD

# Blocks and smaller modules --------------------
class MarchDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        modules = [
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, 3),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(out_channels, out_channels, 2, stride=2),
                nn.LeakyReLU(negative_slope=0.2),
                ]
        self.module = nn.Sequential(*modules)
    def forward(self, batch):
        return self.module(batch)

class MarchUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        modules = [
                nn.Upsample(scale_factor=2),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, 3),
                ]
        self.module = nn.Sequential(*modules)
    def forward(self, batch):
        return self.module(batch)

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

# Encoders ---------------------------------------

Encoder = nn.Sequential(
        TrivialDownBlock(32, 64, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        TrivialDownBlock(64, 128, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        TrivialDownBlock(128, 256, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        TrivialDownBlock(256, 256, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        TrivialDownBlock(256, 256, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(256, 256, 4),
        nn.LeakyReLU(negative_slope=0.2),
        Flatten(256),
        Encoding_layer(256)
        # A sigmoid could be nice here
        )

# Decoders/generators ----------------------------

Generator = nn.Sequential(
        nn.ConvTranspose2d(128, 256, 4),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.LeakyReLU(negative_slope=0.2),
        TrivialUpBlock(256, 256, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        TrivialUpBlock(256, 256, nn.LeakyReLU(negative_slope=0.2)),
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
        TrivialDownBlock(256, 256, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        TrivialDownBlock(256, 256, nn.LeakyReLU(negative_slope=0.2)),
        nn.LeakyReLU(negative_slope=0.2),
        MiniBatchSTD(),
        nn.Conv2d(257, 256, 3, padding=1),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(256, 256, 4),
        nn.LeakyReLU(negative_slope=0.2),
        Flatten(256),
        nn.Linear(256, 1),
        )

# March networks
MarchEncoder = nn.Sequential(
        MarchDownBlock(32, 64),
        MarchDownBlock(64, 128),
        MarchDownBlock(128, 256),
        MarchDownBlock(256, 256),
        MarchDownBlock(256, 256),
        nn.Conv2d(256, 256, 4),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(256,256, 1),
        nn.LeakyReLU(negative_slope=0.2),
        Flatten(256),
        Encoding_layer(256),
        )

MarchDiscriminator = nn.Sequential(
        MarchDownBlock(32, 64),
        MarchDownBlock(64, 128),
        MarchDownBlock(128, 256),
        MarchDownBlock(256, 256),
        MarchDownBlock(256, 256),
        nn.Conv2d(256, 256, 4),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(256, 256, 1),
        nn.LeakyReLU(negative_slope=0.2),
        MiniBatchSTD(),
        Flatten(257),
        nn.Linear(257, 1),
        )

MarchGenerator = nn.Sequential(
        nn.ConvTranspose2d(128, 256, 4),
        nn.LeakyReLU(negative_slope=0.2),
        MarchUpBlock(256, 256),
        MarchUpBlock(256, 256),
        MarchUpBlock(256, 128),
        MarchUpBlock(128, 64),
        MarchUpBlock(64, 32)
        )

# Image to image models --------------------------------------

