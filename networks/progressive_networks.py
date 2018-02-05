import torch
import torch.nn as nn
import torch.nn.functional as F


class TrivialUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.activation = activation
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.thinner = nn.Conv2d(in_channels, out_channels, 3, stride=1)

    def forward(self, feature_map):
        return self.thinner(F.normalize(self.activation(self.upconv(feature_map))))


class TrivialDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.activation = activation
        self.fattener = nn.Conv2d(in_channels, out_channels, 3, stride=1)
        self.conv = nn.Conv2d(in_channels, in_channels, 2, stride=2)

    def forward(self, feature_map):
        return self.conv(self.activation(self.fattener(feature_map)))


class TrivialPGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.inflate = nn.ConvTranspose2d(512, 512, 4)  # Latent -> 4x4
        self.low_conv = nn.Conv2d(512, 512, 3, padding=1)  # Preserve shape

        self.up1 = TrivialUpBlock(512, 256, self.activation)  # 4x4 -> 8x8
        self.up2 = TrivialUpBlock(256, 256, self.activation)  # 8x8 -> 16x16
        self.up3 = TrivialUpBlock(256, 128, self.activation)  # 16x16 -> 32x32
        self.up4 = TrivialUpBlock(128, 64, self.activation)  # 32x32 -> 64x64
        self.up5 = TrivialUpBlock(64, 32, self.activation)  # 64x64 -> 128x128
        self.up6 = TrivialUpBlock(32, 16, self.activation)  # 128x128 -> 256x256

        self.blocks = [self.up1, self.up2, self.up3, self.up4, self.up5, self.up6]

    def forward(self, z, levels=6):
        img = F.normalize(self.activation(self.inflate(z)))
        img = F.normalize(self.activation(self.low_conv(img)))

        for i in range(levels):
            img = F.normalize(self.activation(self.block[i](img)))
        return img


class TrivialPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.low_conv = nn.Conv2d(512, 512, 3, padding=1)
        self.deflate = nn.Conv2d(512, 512, 4)
        self.fc = nn.Linear(512, 1)

        self.down6 = TrivialDownBlock(16, 32, self.activation)  # 256x256 -> 128x128
        self.down5 = TrivialDownBlock(32, 64, self.activation)  # 128x128 -> 64x64
        self.down4 = TrivialDownBlock(64, 128, self.activation)  # 64x64 -> 32x32
        self.down3 = TrivialDownBlock(128, 256, self.activation)  # 32x32 -> 16x16
        self.down2 = TrivialDownBlock(256, 256, self.activation)  # 16x16 -> 8x8
        self.down1 = TrivialDownBlock(256, 512, self.activation)  # 8x8 -> 4x4

        self.blocks = []

    def forward(self, img, levels=6):
        for i in range(levels):
            img = self.activation(self.blocks[i])

        img = self.activation(self.low_conv(img))
        flat = self.activation(self.deflate(img).view(-1, 512))
        return self.fc(flat)


