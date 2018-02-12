import torch
import torch.nn as nn
import torch.nn.functional as F


class TrivialUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.activation = activation
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.thinner = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, feature_map):
        return self.thinner(F.normalize(self.activation(self.upconv(feature_map))))


class UpsamplingUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.activation = activation
        self.thinner = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.thinker = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, feature_map):
        up = F.upsample(feature_map, scale_factor=2)
        img = self.activation(F.normalize(self.thinner(up)))
        return self.activation(F.normalize(self.thinker(img)))


class TrivialDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.activation = activation
        self.fattener = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv = nn.Conv2d(out_channels, out_channels, 2, stride=2)

    def forward(self, feature_map):
        return self.conv(self.activation(self.fattener(feature_map)))


class DownsamplingDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.activation = activation
        self.thinker = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.fattener = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, feature_map):
        img = self.activation(self.thinker(feature_map))
        img = self.activation(self.fattener(img))
        return F.avg_pool2d(img, 2, stride=2)


class TrivialGenerator(nn.Module):
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
            img = F.normalize(self.activation(self.blocks[i](img)))
        return img

    def fade_in(self, z, levels=6):
        img = F.normalize(self.activation(self.inflate(z)))
        img = F.normalize(self.activation(self.low_conv(img)))

        for i in range(levels):
            small = img
            img = F.normalize(self.activation(self.blocks[i](small)))
        return img, small


class TrivialDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.low_conv = nn.Conv2d(512, 512, 3, padding=1)
        self.deflate = nn.Conv2d(512, 512, 4)
        self.fc = nn.Linear(512, 1)

        self.down1 = TrivialDownBlock(16, 32, self.activation)  # 256x256 -> 128x128
        self.down2 = TrivialDownBlock(32, 64, self.activation)  # 128x128 -> 64x64
        self.down3 = TrivialDownBlock(64, 128, self.activation)  # 64x64 -> 32x32
        self.down4 = TrivialDownBlock(128, 256, self.activation)  # 32x32 -> 16x16
        self.down5 = TrivialDownBlock(256, 256, self.activation)  # 16x16 -> 8x8
        self.down6 = TrivialDownBlock(256, 511, self.activation)  # 8x8 -> 4x4

        self.blocks = [self.down1, self.down2, self.down3, self.down4, self.down5, self.down6]

    def forward(self, img, levels=6):
        start = 6 - levels
        for i in range(levels):
            img = self.activation(self.blocks[start + i](img))

        # Minibatch stddev
        minibatch_std = img.std(0).mean().expand(img.shape[0], 1, 4, 4)
        img = torch.cat([img, minibatch_std], dim=1)

        # Cont
        img = self.activation(self.low_conv(img))
        flat = self.activation(self.deflate(img).view(-1, 512))
        return self.fc(flat)

    def fade_in(self, img, small, alpha, levels=6):
        start = 6 - levels
        img = self.activation(self.blocks[start](img))
        start += 1
        levels -= 1
        img = alpha * img + (1 - alpha) * small
        for i in range(levels):
            img = self.activation(self.blocks[start + i](img))

        # Minibatch stddev
        minibatch_std = img.std(0).mean().expand(img.shape[0], 1, 4, 4)
        img = torch.cat([img, minibatch_std], dim=1)

        # Cont
        img = self.activation(self.low_conv(img))
        flat = self.activation(self.deflate(img).view(-1, 512))
        return self.fc(flat)


# Light networks --------------------------------------------

class TrivialGeneratorLight(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.inflate = nn.ConvTranspose2d(128, 128, 4)  # Latent -> 4x4
        self.low_conv = nn.Conv2d(128, 128, 3, padding=1)  # Preserve shape

        self.up1 = TrivialUpBlock(128, 112, self.activation)  # 4x4 -> 8x8
        self.up2 = TrivialUpBlock(112, 96, self.activation)  # 8x8 -> 16x16
        self.up3 = TrivialUpBlock(96, 80, self.activation)  # 16x16 -> 32x32
        self.up4 = TrivialUpBlock(80, 64, self.activation)  # 32x32 -> 64x64
        self.up5 = TrivialUpBlock(64, 32, self.activation)  # 64x64 -> 128x128
        self.up6 = TrivialUpBlock(32, 16, self.activation)  # 128x128 -> 256x256

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


class TrivialDiscriminatorLight(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.low_conv = nn.Conv2d(128, 128, 3, padding=1)
        self.deflate = nn.Conv2d(128, 128, 4)
        self.fc = nn.Linear(128, 1)

        self.down1 = TrivialDownBlock(16, 32, self.activation)  # 256x256 -> 128x128
        self.down2 = TrivialDownBlock(32, 64, self.activation)  # 128x128 -> 64x64
        self.down3 = TrivialDownBlock(64, 80, self.activation)  # 64x64 -> 32x32
        self.down4 = TrivialDownBlock(80, 96, self.activation)  # 32x32 -> 16x16
        self.down5 = TrivialDownBlock(96, 112, self.activation)  # 16x16 -> 8x8
        self.down6 = TrivialDownBlock(112, 127, self.activation)  # 8x8 -> 4x4

        self.blocks = [self.down1, self.down2, self.down3, self.down4, self.down5, self.down6]

    def forward(self, img, levels=6):
        start = 6 - levels
        for i in range(levels):
            img = self.activation(self.blocks[start + i](img))

        # Minibatch stddev
        minibatch_std = img.std(0).mean().expand(img.shape[0], 1, 4, 4)
        img = torch.cat([img, minibatch_std], dim=1)

        # Cont
        img = self.activation(self.low_conv(img))
        flat = self.activation(self.deflate(img).view(-1, 128))
        return self.fc(flat)

    def fade_in(self, img, small, alpha, levels=6):
        start = 6 - levels
        img = self.activation(self.blocks[start](img))
        start += 1
        levels -= 1
        img = alpha * img + (1 - alpha) * small
        for i in range(levels):
            img = self.activation(self.blocks[start + i](img))

        # Minibatch stddev
        minibatch_std = img.std(0).mean().expand(img.shape[0], 1, 4, 4)
        img = torch.cat([img, minibatch_std], dim=1)

        # Cont
        img = self.activation(self.low_conv(img))
        flat = self.activation(self.deflate(img).view(-1, 128))
        return self.fc(flat)


# Sampling networks --------------------------------------------------------------

class SamplingGeneratorLight(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.inflate = nn.ConvTranspose2d(128, 128, 4)  # Latent -> 4x4
        self.low_conv = nn.Conv2d(128, 128, 3, padding=1)  # Preserve shape

        self.up1 = UpsamplingUpBlock(128, 112, self.activation)  # 4x4 -> 8x8
        self.up2 = UpsamplingUpBlock(112, 96, self.activation)  # 8x8 -> 16x16
        self.up3 = UpsamplingUpBlock(96, 80, self.activation)  # 16x16 -> 32x32
        self.up4 = UpsamplingUpBlock(80, 64, self.activation)  # 32x32 -> 64x64
        self.up5 = UpsamplingUpBlock(64, 32, self.activation)  # 64x64 -> 128x128
        self.up6 = UpsamplingUpBlock(32, 16, self.activation)  # 128x128 -> 256x256

        self.blocks = [self.up1, self.up2, self.up3, self.up4, self.up5, self.up6]

    def forward(self, z, levels=6):
        img = self.activation(self.inflate(z))
        img = self.activation(F.normalize(self.low_conv(img)))

        for i in range(levels):
            img = self.blocks[i](img)
        return img

    def fade_in(self, z, levels=6):
        img = F.normalize(self.activation(self.inflate(z)))
        img = F.normalize(self.activation(self.low_conv(img)))

        for i in range(levels):
            small = img
            img = self.blocks[i](small)
        return img, small


class SamplingDiscriminatorLight(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.low_conv = nn.Conv2d(128, 128, 3, padding=1)
        self.deflate = nn.Conv2d(128, 128, 4)
        self.fc = nn.Linear(128, 1)

        self.down1 = DownsamplingDownBlock(16, 32, self.activation)  # 256x256 -> 128x128
        self.down2 = DownsamplingDownBlock(32, 64, self.activation)  # 128x128 -> 64x64
        self.down3 = DownsamplingDownBlock(64, 80, self.activation)  # 64x64 -> 32x32
        self.down4 = DownsamplingDownBlock(80, 96, self.activation)  # 32x32 -> 16x16
        self.down5 = DownsamplingDownBlock(96, 112, self.activation)  # 16x16 -> 8x8
        self.down6 = DownsamplingDownBlock(112, 127, self.activation)  # 8x8 -> 4x4

        self.blocks = [self.down1, self.down2, self.down3, self.down4, self.down5, self.down6]

    def forward(self, img, levels=6):
        start = 6 - levels
        for i in range(levels):
            img = self.blocks[start + i](img)

        # Minibatch stddev
        minibatch_std = img.std(0).mean().expand(img.shape[0], 1, 4, 4)
        img = torch.cat([img, minibatch_std], dim=1)

        # Cont
        img = self.activation(self.low_conv(img))
        flat = self.activation(self.deflate(img).view(-1, 128))
        return self.fc(flat)

    def fade_in(self, img, small, alpha, levels=6):
        start = 6 - levels
        img = self.activation(self.blocks[start](img))
        start += 1
        levels -= 1
        img = alpha * img + (1 - alpha) * small
        for i in range(levels):
            img = self.activation(self.blocks[start + i](img))

        # Minibatch stddev
        minibatch_std = img.std(0).mean().expand(img.shape[0], 1, 4, 4)
        img = torch.cat([img, minibatch_std], dim=1)

        # Cont
        img = self.activation(self.low_conv(img))
        flat = self.activation(self.deflate(img).view(-1, 128))
        return self.fc(flat)
