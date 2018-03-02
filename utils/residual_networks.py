import torch
import torch.nn as nn
import torch.nn.functional as F


class ResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.activation = activation
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.thinner = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.resthin = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, feature_map):
        up = F.upsample(feature_map, scale_factor=2)
        return self.thinner(F.normalize(self.activation(self.upconv(feature_map)))) + self.resthin(up)


class ResDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.activation = activation
        self.fattener = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.resfat = nn.Conv2d(in_channels, out_channels, 1)
        self.conv = nn.Conv2d(out_channels, out_channels, 2, stride=2)

    def forward(self, feature_map):
        down = F.avg_pool2d(feature_map, 2, stride=2)
        return self.conv(F.normalize(self.activation(self.fattener(feature_map)))) + self.resfat(down)


# Networks --------------------------------------------

class ResGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.inflate = nn.ConvTranspose2d(128, 128, 4)  # Latent -> 4x4
        self.low_conv = nn.Conv2d(128, 128, 3, padding=1)  # Preserve shape

        self.up1 = ResUpBlock(128, 112, self.activation)  # 4x4 -> 8x8
        self.up2 = ResUpBlock(112, 96, self.activation)  # 8x8 -> 16x16
        self.up3 = ResUpBlock(96, 80, self.activation)  # 16x16 -> 32x32
        self.up4 = ResUpBlock(80, 64, self.activation)  # 32x32 -> 64x64
        self.up5 = ResUpBlock(64, 32, self.activation)  # 64x64 -> 128x128
        self.up6 = ResUpBlock(32, 16, self.activation)  # 128x128 -> 256x256

        self.blocks = [self.up1, self.up2, self.up3, self.up4, self.up5, self.up6]

    def forward(self, z, levels=6):
        img = F.normalize(self.activation(self.inflate(z)))
        img = F.normalize(self.activation(self.low_conv(img)))

        for i in range(levels):
            img = F.normalize(self.activation(self.blocks[i](img)))
        return img

    def fade_in(self, z, levels=6):
        img = F.normalize(self.activation(self.inflate(z)))
        img = F.normalize(self.activation(self.low_conv(img)))

        for i in range(levels):
            small = img
            img = F.normalize(self.activation(self.blocks[i](small)))
        return img, small


class ResDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.low_conv = nn.Conv2d(128, 128, 3, padding=1)
        self.deflate = nn.Conv2d(128, 128, 4)
        self.fc = nn.Linear(128, 1)

        self.down1 = ResDownBlock(16, 32, self.activation)  # 256x256 -> 128x128
        self.down2 = ResDownBlock(32, 64, self.activation)  # 128x128 -> 64x64
        self.down3 = ResDownBlock(64, 80, self.activation)  # 64x64 -> 32x32
        self.down4 = ResDownBlock(80, 96, self.activation)  # 32x32 -> 16x16
        self.down5 = ResDownBlock(96, 112, self.activation)  # 16x16 -> 8x8
        self.down6 = ResDownBlock(112, 127, self.activation)  # 8x8 -> 4x4

        self.blocks = [self.down1, self.down2, self.down3, self.down4, self.down5, self.down6]

    def forward(self, img, levels=6):
        start = 6 - levels
        for i in range(levels):
            img = F.normalize(self.activation(self.blocks[start + i](img)))

        # Minibatch stddev
        minibatch_std = img.std(0).mean().expand(img.shape[0], 1, 4, 4)
        img = torch.cat([img, minibatch_std], dim=1)

        # Cont
        img = self.activation(self.low_conv(img))
        flat = self.activation(self.deflate(img).view(-1, 128))
        return self.fc(flat)


class ResEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.low_conv = nn.Conv2d(128, 128, 3, padding=1)
        self.deflate = nn.Conv2d(128, 128, 4)
        self.fc = nn.Linear(128, 128)

        self.down1 = ResDownBlock(16, 32, self.activation)  # 256x256 -> 128x128
        self.down2 = ResDownBlock(32, 64, self.activation)  # 128x128 -> 64x64
        self.down3 = ResDownBlock(64, 80, self.activation)  # 64x64 -> 32x32
        self.down4 = ResDownBlock(80, 96, self.activation)  # 32x32 -> 16x16
        self.down5 = ResDownBlock(96, 112, self.activation)  # 16x16 -> 8x8
        self.down6 = ResDownBlock(112, 128, self.activation)  # 8x8 -> 4x4

        self.blocks = [self.down1, self.down2, self.down3, self.down4, self.down5, self.down6]

    def forward(self, img, levels=6):
        start = 6 - levels
        for i in range(levels):
            img = F.normalize(self.activation(self.blocks[start + i](img)))

        # Cont
        img = self.activation(self.low_conv(img))
        flat = self.activation(self.deflate(img).view(-1, 128))
        return self.fc(flat), None


