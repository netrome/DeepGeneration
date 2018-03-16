import torch.nn as nn
import torch.nn.functional as F
import torch

from .progressive_networks import TrivialDownBlock, DownsamplingDownBlock


# Encoders ---------------------------------------
class TrivialEncoderLight(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.low_conv = nn.Conv2d(128, 128, 3, padding=1)
        self.deflate = nn.Conv2d(128, 128, 4)
        self.to_mean = nn.Linear(128, 128)
        self.to_log_var = nn.Linear(128, 128)

        self.down1 = TrivialDownBlock(16, 32, self.activation)  # 256x256 -> 128x128
        self.down2 = TrivialDownBlock(32, 64, self.activation)  # 128x128 -> 64x64
        self.down3 = TrivialDownBlock(64, 80, self.activation)  # 64x64 -> 32x32
        self.down4 = TrivialDownBlock(80, 96, self.activation)  # 32x32 -> 16x16
        self.down5 = TrivialDownBlock(96, 112, self.activation)  # 16x16 -> 8x8
        self.down6 = TrivialDownBlock(112, 128, self.activation)  # 8x8 -> 4x4

        self.blocks = [self.down1, self.down2, self.down3, self.down4, self.down5, self.down6]

    def forward(self, img, levels=5):
        start = 6 - levels
        for i in range(levels):
            img = self.activation(self.blocks[start + i](img))

        img = self.activation(self.low_conv(img))
        flat = self.activation(self.deflate(img).view(-1, 128))
        return self.to_mean(flat), self.to_log_var(flat)


class SamplingEncoderLight(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.low_conv = nn.Conv2d(128, 128, 3, padding=1)
        self.deflate = nn.Conv2d(128, 128, 4)
        self.to_mean = nn.Linear(128, 128)
        self.to_log_var = nn.Linear(128, 128)

        self.down1 = DownsamplingDownBlock(16, 32, self.activation)  # 256x256 -> 128x128
        self.down2 = DownsamplingDownBlock(32, 64, self.activation)  # 128x128 -> 64x64
        self.down3 = DownsamplingDownBlock(64, 80, self.activation)  # 64x64 -> 32x32
        self.down4 = DownsamplingDownBlock(80, 96, self.activation)  # 32x32 -> 16x16
        self.down5 = DownsamplingDownBlock(96, 112, self.activation)  # 16x16 -> 8x8
        self.down6 = DownsamplingDownBlock(112, 128, self.activation)  # 8x8 -> 4x4

        self.blocks = [self.down1, self.down2, self.down3, self.down4, self.down5, self.down6]

    def forward(self, img, levels=6):
        start = 6 - levels
        for i in range(levels):
            img = self.activation(self.blocks[start + i](img))

        img = self.activation(self.low_conv(img))
        flat = self.activation(self.deflate(img).view(-1, 128))
        return self.to_mean(flat), self.to_log_var(flat)


# Decoders/generators ----------------------------


# Image to image models --------------------------------------
class ImageToImage(nn.Module):
    """ For semantic segmentation """
    def __init__(self):
        super().__init__()

        self.fromImg = nn.Conv2d(1, 16, 1)
        self.toImg = nn.Conv2d(16, 1, 1)

        self.activation = nn.ReLU()

        self.down1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 256 -> 128
        self.down2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 128 -> 64
        self.down3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 64 -> 32
        self.down4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)  # 32 -> 16
        self.down5 = nn.Conv2d(256, 256, 3, stride=2, padding=1)  # 16 -> 8
        self.down6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)  # 8 -> 4

        self.up1 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.up4 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.up5 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.up6 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)

        self.downs = [self.down1, self.down2, self.down3, self.down4, self.down5, self.down6]
        self.ups = [self.up1, self.up2, self.up3, self.up4, self.up5, self.up6]

    def forward(self, batch):
        map = self.fromImg(batch)
        maps = [0 for i in range(6)]
        for i in range(6):
            maps[5 - i] = map
            map = self.activation(self.downs[i](map))

        # Do stuff with feature map - but not now

        for i in range(6):
            map = self.activation(self.ups[i](map) + maps[i])

        return self.toImg(map)

