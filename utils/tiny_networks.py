import torch.nn as nn
import torch.nn.functional as F
import torch

def refpad(x, pad):
    return F.pad(x, [0, 0, pad, pad], mode="reflect")

class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, padding=0, stride=1):
        super().__init__()
        self.pad = nn.ReflectionPad2d(padding)

        self.depthwise = nn.Conv2d(in_channels, in_channels, filter_size,
                stride=stride, groups=in_channels)
        self.widthwise_fc = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, batch):
        return self.widthwise_fc(self.depthwise(self.pad(batch)))

class DSConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, padding=0, stride=1):
        super().__init__()
        self.pad = nn.ReflectionPad2d(padding)

        self.widthwise_fc = nn.Conv2d(in_channels, out_channels, 1)
        self.depthwise = nn.ConvTranspose2d(out_channels, out_channels, 
                filter_size, stride=stride, groups=out_channels)

    def forward(self, batch):
        return self.depthwise(self.pad(self.widthwise_fc(batch)))

class DSDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.activation = nn.LeakyReLU()
        self.think = DSConv(in_channels, out_channels, 3, padding=1)
        self.down = DSConv(out_channels, out_channels, 2, stride=2)
    
    def forward(self, batch):
        return self.down(self.activation(self.think(batch)))

class DSUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.activation = nn.LeakyReLU()
        self.up = DSConvTranspose(in_channels, in_channels, 2, stride=2)
        self.think = DSConv(in_channels, out_channels, 3, padding=1)
    
    def forward(self, batch):
        return self.think(self.activation(F.normalize(self.up(batch))))


# Encoders ---------------------------------------
class DSEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU()

        self.down1 = DSConv(16, 32, 4, padding=1, stride=2)
        self.down2 = DSConv(32, 64, 4, padding=1, stride=2)
        self.down3 = DSConv(64, 80, 4, padding=1, stride=2)
        self.down4 = DSConv(80, 96, 4, padding=1, stride=2)
        self.down5 = DSConv(96, 112, 4, padding=1, stride=2)
        self.down6 = DSConv(112, 128, 4, padding=1, stride=2)

        self.deflate = nn.Conv2d(128, 128, 4)

        self.low_think = DSConv(128, 128, 3, padding=1)

        self.to_mean = nn.Linear(128, 128)

    def forward(self, img):
        img = self.activation(self.down1(img))
        img = self.activation(self.down2(img))
        img = self.activation(self.down3(img))
        img = self.activation(self.down4(img))
        img = self.activation(self.down5(img))
        img = self.activation(self.down6(img))
        img = self.activation(self.low_think(img))
        vec = self.deflate(img)
        return self.to_mean(vec.view(-1, 128)), None

# Decoders/generators ----------------------------
class DSDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU()

        self.inflate = nn.ConvTranspose2d(128, 128, 4, groups=128) 
        self.low_think = DSConv(128, 128, 3, padding=1)

        self.up1 = DSConvTranspose(128, 112, 2, padding=0, stride=2)
        self.up2 = DSConvTranspose(112, 96, 2, padding=0, stride=2)
        self.up3 = DSConvTranspose(96, 80, 2, padding=0, stride=2)
        self.up4 = DSConvTranspose(80, 64, 2, padding=0, stride=2)
        self.up5 = DSConvTranspose(64, 32, 2, padding=0, stride=2)
        self.up6 = DSConvTranspose(32, 16, 2, padding=0, stride=2)

    def forward(self, latent):
        img = F.normalize(self.activation(self.inflate(latent)))
                          
        img = F.normalize(self.activation(self.low_think(img)))
        img = F.normalize(self.activation(self.up1(img)))
        img = F.normalize(self.activation(self.up2(img)))
        img = F.normalize(self.activation(self.up3(img)))
        img = F.normalize(self.activation(self.up4(img)))
        img = F.normalize(self.activation(self.up5(img)))
        img = F.normalize(self.activation(self.up6(img)))
        return img


# Discriminators ---------------------------------
class DSDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU()

        self.down1 = DSConv(16, 32, 4, padding=1, stride=2)
        self.down2 = DSConv(32, 64, 4, padding=1, stride=2)
        self.down3 = DSConv(64, 80, 4, padding=1, stride=2)
        self.down4 = DSConv(80, 96, 4, padding=1, stride=2)
        self.down5 = DSConv(96, 112, 4, padding=1, stride=2)
        self.down6 = DSConv(112, 128, 4, padding=1, stride=2)

        self.deflate = nn.Conv2d(128, 128, 4)

        self.low_think = DSConv(128, 128, 3, padding=1)

        self.to_pred = nn.Linear(128, 1)

    def forward(self, img):
        img = self.activation(self.down1(img))
        img = self.activation(self.down2(img))
        img = self.activation(self.down3(img))
        img = self.activation(self.down4(img))
        img = self.activation(self.down5(img))
        img = self.activation(self.down6(img))

        #minibatch_std = img.std(0).mean().expand(img.shape[0], 1, 4, 4)
        #img = torch.cat([img, minibatch_std], dim=1)

        img = self.activation(self.low_think(img))
        vec = self.deflate(img)

        return self.to_pred(vec.view(-1, 128))

