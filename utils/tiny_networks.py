import torch.nn as nn
import torch.nn.functional as F
import torch

def refpad(x, pad):
    return F.pad(x, pad, mode="reflect")

class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, padding=0, stride=1):
        super().__init__()
        self.pad = padding

        self.depthwise = nn.Conv2d(in_channels, in_channels, filter_size,
                stride=stride, groups=in_channels)
        self.widthwise_fc = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, batch):
        return self.widthwise_fc(self.depthwise(refpad(batch, self.pad)))

class DSConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, padding=0, stride=1):
        super().__init__()
        self.pad = padding

        self.widthwise_fc = nn.Conv2d(in_channels, out_channels, 1)
        self.depthwise = nn.ConvTranspose2d(out_channels, out_channels, 
                filter_size, stride=stride, groups=out_channels)

    def forward(self, batch):
        return self.depthwise(refpad(self.widthwise_fc(batch), self.pad))


# Encoders ---------------------------------------
class DSEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DSConv(16, 32, 4, padding=2, stride=4)
        self.down2 = DSConv(32, 64, 4, padding=2, stride=4)
        self.down3 = DSConv(64, 128, 4, padding=2, stride=4)

        self.deflate = nn.Conv2d(128, 128, 4)

        self.to_mean = nn.Linear(128, 128)

    def forward(self, img):
        img = F.leaky_relu(self.down1(img), 0.2)
        img = F.leaky_relu(self.down2(img), 0.2)
        img = F.leaky_relu(self.down3(img), 0.2)
        vec = self.deflate(img)
        return self.to_mean(vec.view(-1, 128)), None

# Decoders/generators ----------------------------
class DSDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.inflate = nn.ConvTranspose2d(128, 128, 4) # Latent -> 4x4

        self.up1 = DSConvTranspose(128, 64, 4, padding=2, stride=4) #4x4->16x16
        self.up2 = DSConvTranspose(64, 32, 4, padding=2, stride=4) #16x16->64x64
        self.up3 = DSConvTranspose(32, 16, 4, padding=2, stride=4) #64x64->256x256

    def forward(self, latent):
        img = F.leaky_relu(F.normalize(self.inflate(latent)), 0.2)
        img = F.leaky_relu(F.normalize(self.up1(img)), 0.2)
        img = F.leaky_relu(F.normalize(self.up2(img)), 0.2)
        img = F.leaky_relu(F.normalize(self.up3(img)), 0.2)
        return img


# Discriminators ---------------------------------
class DSDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DSConv(16, 32, 4, padding=2, stride=4)
        self.down2 = DSConv(32, 64, 4, padding=2, stride=4)
        self.down3 = DSConv(64, 128, 4, padding=2, stride=4)

        self.deflate = nn.Conv2d(128, 128, 4)

        self.to_pred = nn.Linear(128, 128)

    def forward(self, img):
        img = F.leaky_relu(self.down1(img), 0.2)
        img = F.leaky_relu(self.down2(img), 0.2)
        img = F.leaky_relu(self.down3(img), 0.2)
        vec = self.deflate(img)
        return self.to_pred(vec.view(-1, 128))

