import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
    def forward(self, batch):
        return batch.view(-1, self.size)

class Encode(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, batch):
        return batch, None

class MiniBatchSTD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        shape = list(img.shape)
        shape[1] = 1
        minibatch_std = img.std(0).mean().expand(shape) + 1e-6
        out = torch.cat([img, minibatch_std], dim=1)
        return out


NoNormGenerator = nn.Sequential(
        nn.ConvTranspose2d(42, 128, 4),
        nn.SELU(),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.SELU(),
        nn.ConvTranspose2d(128, 112, 2, stride=2),
        nn.SELU(),
        nn.Conv2d(112, 112, 3, padding=1),
        nn.SELU(),
        nn.Conv2d(112, 112, 3, padding=1),
        nn.SELU(),
        nn.ConvTranspose2d(112, 96, 2, stride=2),
        nn.SELU(),
        nn.Conv2d(96, 96, 3, padding=1),
        nn.SELU(),
        nn.Conv2d(96, 96, 3, padding=1),
        nn.SELU(),
        nn.ConvTranspose2d(96, 80, 2, stride=2),
        nn.SELU(),
        nn.Conv2d(80, 80, 3, padding=1),
        nn.SELU(),
        nn.Conv2d(80, 80, 3, padding=1),
        nn.SELU(),
        nn.ConvTranspose2d(80, 64, 2, stride=2),
        nn.SELU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.SELU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.SELU(),
        nn.ConvTranspose2d(64, 32, 2, stride=2),
        nn.SELU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.SELU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.SELU(),
        nn.ConvTranspose2d(32, 16, 2, stride=2),
        nn.SELU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(16, 16, 3),
        nn.SELU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(16, 16, 3),
        nn.Tanh()  # Prevent signal magnitude escalation
        )

NoNormDiscriminator = nn.Sequential(
        nn.Conv2d(16, 16, 3, padding=1),
        nn.SELU(),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.SELU(),
        nn.Conv2d(16, 32, 2, stride=2),
        nn.SELU(),
        nn.Conv2d(32, 64, 2, stride=2),
        nn.SELU(),
        nn.Conv2d(64, 80, 2, stride=2),
        nn.SELU(),
        nn.Conv2d(80, 96, 2, stride=2),
        nn.SELU(),
        nn.Conv2d(96, 112, 2, stride=2),
        nn.SELU(),
        nn.Conv2d(112, 128, 2, stride=2),
        nn.SELU(),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.SELU(),
        MiniBatchSTD(),
        nn.Conv2d(129, 42, 4),
        nn.SELU(),
        Flatten(42),
        nn.Linear(42, 1)
        )

NoNormEncoder = nn.Sequential(
        nn.Conv2d(16, 16, 3, padding=1),
        nn.SELU(),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.SELU(),
        nn.Conv2d(16, 32, 2, stride=2),
        nn.SELU(),
        nn.Conv2d(32, 64, 2, stride=2),
        nn.SELU(),
        nn.Conv2d(64, 80, 2, stride=2),
        nn.SELU(),
        nn.Conv2d(80, 96, 2, stride=2),
        nn.SELU(),
        nn.Conv2d(96, 112, 2, stride=2),
        nn.SELU(),
        nn.Conv2d(112, 128, 2, stride=2),
        nn.SELU(),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.SELU(),
        nn.Conv2d(128, 42, 4),
        nn.SELU(),
        Flatten(42),
        Encode(),
        )

