import torch.nn as nn
import torch.nn.functional as F
import torch

from .progressive_networks import TrivialDownBlock, TrivialUpBlock


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

    def forward(self, img, levels=6):
        start = 6 - levels
        for i in range(levels):
            img = self.activation(self.blocks[start + i](img))

        img = self.activation(self.low_conv(img))
        flat = self.activation(self.deflate(img).view(-1, 128))
        return self.to_mean(flat), self.to_log_var(flat)

# Decoders/generators ----------------------------

