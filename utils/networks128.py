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
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels, out_channels, 4, stride=2),
                nn.LeakyReLU(negative_slope=0.2),
                ]
        self.module = nn.Sequential(*modules)
    def forward(self, batch):
        return self.module(batch)

class MarchUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True, normalize=False):
        super().__init__()
        up = nn.Upsample(scale_factor=2) if upsample else nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        modules = [
                up,
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, in_channels, 3),
                nn.LeakyReLU(negative_slope=0.2),
                Normalize(normalize),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, 3),
                nn.LeakyReLU(negative_slope=0.2),
                Normalize(normalize),
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

class Normalize(nn.Module):
    def __init__(self, normalize=True):
        super().__init__()
        self.do_it = normalize
    def forward(self, batch):
        if self.do_it:
            return F.normalize(batch)
        else:
            return batch

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
        Normalize(),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.LeakyReLU(negative_slope=0.2),
        Normalize(),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.LeakyReLU(negative_slope=0.2),
        Normalize(),
        MarchUpBlock(256, 256, normalize=True),
        MarchUpBlock(256, 256, normalize=True),
        MarchUpBlock(256, 128, normalize=True),
        MarchUpBlock(128, 64, normalize=True, upsample=False),
        MarchUpBlock(64, 32, upsample=False)
        )

# March 2 networks ----------------------------------------------------
def March2ConvBlock(in_channels, out_channels, kernel_size, stride=1, padding=0, normalize=False):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(negative_slope=0.2),
            Normalize(normalize)
            )

def March2ConvTransposeBlock(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride),
            nn.LeakyReLU(negative_slope=0.2),
            Normalize()
            )

MarchGenerator2 = nn.Sequential(
        March2ConvTransposeBlock(128, 512, 4),
        March2ConvBlock(512, 512, 3, padding=1),
        March2ConvBlock(512, 512, 3, padding=1),
        March2ConvTransposeBlock(512, 256, 2, stride=2),
        March2ConvBlock(256, 256, 3, padding=1),
        March2ConvTransposeBlock(256, 256, 2, stride=2),
        March2ConvBlock(256, 256, 3, padding=1),
        March2ConvTransposeBlock(256, 128, 2, stride=2),
        March2ConvBlock(128, 128, 3, padding=1),
        March2ConvTransposeBlock(128, 64, 2, stride=2),
        March2ConvBlock(64, 64, 3, padding=1),
        March2ConvTransposeBlock(64, 32, 2, stride=2),
        March2ConvBlock(32, 32, 3, padding=1),
        )

MarchDiscriminator2 = nn.Sequential(
        March2ConvBlock(32, 64, 2, stride=2),
        March2ConvBlock(64, 128, 2, stride=2),
        March2ConvBlock(128, 256, 2, stride=2),
        March2ConvBlock(256, 256, 2, stride=2),
        March2ConvBlock(256, 512, 2, stride=2),

        March2ConvBlock(512, 512, 3, padding=1),
        March2ConvBlock(512, 512, 3, padding=1),
        March2ConvBlock(512, 512, 4),
        March2ConvBlock(512, 512, 1),
        MiniBatchSTD(),
        Flatten(513),
        nn.Linear(513, 1),
        )

MarchEncoder2 = nn.Sequential(
        March2ConvBlock(32, 64, 2, stride=2),
        March2ConvBlock(64, 128, 2, stride=2),
        March2ConvBlock(128, 256, 2, stride=2),
        March2ConvBlock(256, 256, 2, stride=2),
        March2ConvBlock(256, 512, 2, stride=2),

        March2ConvBlock(512, 512, 3, padding=1),
        March2ConvBlock(512, 512, 3, padding=1),
        March2ConvBlock(512, 512, 4),
        March2ConvBlock(512, 512, 1),
        Flatten(512),
        Encoding_layer(512),
        )


# Image to image models --------------------------------------

