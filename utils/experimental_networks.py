import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.activation = activation

        self.up1 = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.thinker = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.thinner = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, feature_map):
        up = F.upsample(feature_map, scale_factor=4)
        paramup = F.normalize(self.activation(self.up2(F.normalize(self.activation(self.up1(feature_map))))))
        return self.thinner(F.normalize(self.activation(self.thinker(paramup) + up)))


class ExpDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.activation = activation

        self.down1 = nn.Conv2d(in_channels, in_channels, 2, stride=2)
        self.down2 = nn.Conv2d(in_channels, in_channels, 2, stride=2)

        self.thinker = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.fattener = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, feature_map):
        down = F.avg_pool2d(feature_map, 4, stride=4)
        paramdown = self.activation(self.down2(self.activation(self.down1(feature_map))))
        return self.fattener(self.activation(self.thinker(paramdown) + down))


class ResBlock(nn.Module):
    def __init__(self, channels, activation):
        super().__init__()
        self.activation = activation
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, batch):
        return self.activation(self.conv2(self.activation(self.conv1(batch))) + batch)


# Networks --------------------------------------------

class ExpGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.inflate = nn.ConvTranspose2d(128, 128, 4)  # Latent -> 4x4
        self.low_conv = nn.Conv2d(128, 128, 3, padding=1)  # Preserve shape

        self.res1 = ResBlock(128, self.activation)
        self.res2 = ResBlock(128, self.activation)
        self.res3 = ResBlock(128, self.activation)

        self.norm1 = nn.InstanceNorm2d(128)
        self.norm2 = nn.InstanceNorm2d(128)
        self.norm3 = nn.InstanceNorm2d(128)

        self.up1 = nn.ConvTranspose2d(128, 112, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(112, 96, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(96, 80, 2, stride=2)
        self.up4 = nn.ConvTranspose2d(80, 64, 2, stride=2)
        self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.up6 = nn.ConvTranspose2d(32, 16, 2, stride=2)

        self.ups = [self.up1, self.up2, self.up3, self.up4, self.up5, self.up6]

    def forward(self, z):
        img = F.normalize(self.activation(self.inflate(z)))
        img = F.normalize(self.activation(self.low_conv(img)))

        img = self.norm1(self.activation(self.res1(img)))
        img = self.norm2(self.activation(self.res2(img)))
        img = self.norm3(self.activation(self.res3(img)))

        for up in self.ups:
            img = self.activation(up(img))

        return img


class ExpDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.low_conv = nn.Conv2d(129, 128, 3, padding=1)
        self.deflate = nn.Conv2d(128, 128, 4)
        self.fc = nn.Linear(128, 1)

        self.down1 = nn.Conv2d(16, 32, 2, stride=2)
        self.down2 = nn.Conv2d(32, 64, 2, stride=2)
        self.down3 = nn.Conv2d(64, 80, 2, stride=2)
        self.down4 = nn.Conv2d(80, 96, 2, stride=2)
        self.down5 = nn.Conv2d(96, 112, 2, stride=2)
        self.down6 = nn.Conv2d(112, 128, 2, stride=2)

        self.res1 = ResBlock(128, self.activation)
        self.res2 = ResBlock(128, self.activation)
        self.res3 = ResBlock(128, self.activation)

        self.norm1 = nn.InstanceNorm2d(128)
        self.norm2 = nn.InstanceNorm2d(128)
        self.norm3 = nn.InstanceNorm2d(128)

        self.blocks = [self.down1, self.down2, self.down3, self.down4, self.down5, self.down6]

    def forward(self, img):
        for block in self.blocks:
            img = self.activation(block(img))

        img = self.norm1(self.activation(self.res1(img)))
        img = self.norm2(self.activation(self.res2(img)))
        img = self.norm3(self.activation(self.res3(img)))

        # Minibatch stddev
        minibatch_std = img.std(0).mean().expand(img.shape[0], 1, 4, 4)
        img = torch.cat([img, minibatch_std], dim=1)

        # Cont
        img = self.activation(self.low_conv(img))
        flat = self.activation(self.deflate(img).view(-1, 128))
        return self.fc(flat)


class ExpEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.low_conv = nn.Conv2d(128, 128, 3, padding=1)
        self.deflate = nn.Conv2d(128, 128, 4)
        self.fc = nn.Linear(128, 128)

        self.down1 = nn.Conv2d(16, 32, 2, stride=2)
        self.down2 = nn.Conv2d(32, 64, 2, stride=2)
        self.down3 = nn.Conv2d(64, 80, 2, stride=2)
        self.down4 = nn.Conv2d(80, 96, 2, stride=2)
        self.down5 = nn.Conv2d(96, 112, 2, stride=2)
        self.down6 = nn.Conv2d(112, 128, 2, stride=2)

        self.res1 = ResBlock(128, self.activation)
        self.res2 = ResBlock(128, self.activation)
        self.res3 = ResBlock(128, self.activation)

        self.norm1 = nn.InstanceNorm2d(128)
        self.norm2 = nn.InstanceNorm2d(128)
        self.norm3 = nn.InstanceNorm2d(128)

        self.blocks = [self.down1, self.down2, self.down3, self.down4, self.down5, self.down6]

    def forward(self, img):
        for block in self.blocks:
            img = self.activation(block(img))

        img = self.norm1(self.activation(self.res1(img)))
        img = self.norm2(self.activation(self.res2(img)))
        img = self.norm3(self.activation(self.res3(img)))

        # Minibatch stddev
        #minibatch_std = img.std(0).mean().expand(img.shape[0], 1, 4, 4)
        #img = torch.cat([img, minibatch_std], dim=1)

        # Cont
        img = self.activation(self.low_conv(img))
        flat = self.activation(self.deflate(img).view(-1, 128))
        return self.fc(flat), None

