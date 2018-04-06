import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as data
import torch.utils.data.dataloader as loader
import torchvision.transforms as trans

train_data = data.MNIST("~/Data/mnist/", train=True, transform=trans.ToTensor())
test_data = data.MNIST("~/Data/mnist/", train=False, transform=trans.ToTensor())

latent_size = 20

def get_data_loader(train=True):
    data = train_data if train else test_data
    return loader.DataLoader(data, shuffle=train, pin_memory=True, batch_size=64, drop_last=True)

def get_mean_images():
    raise DeprecationWarning("No more mean images")
    means = torch.zeros(10, 1, 28, 28)
    counts = torch.zeros(10)
    for i, (img, label) in enumerate(train_data):
        means[int(label)] += img
        counts[int(label)] += 1
    means /= counts.view(10, 1, 1, 1)
    return torch.round(means * 1.4)

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.size = args
    def forward(self, batch):
        return batch.view(*self.size)

class ScaledTanh(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, batch):
        return F.tanh(batch) * 10

class MiniBatchSTD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        shape = list(img.shape)
        shape[1] = 1
        minibatch_std = img.std(0).mean().expand(shape) + 1e-6
        out = torch.cat([img, minibatch_std], dim=1)
        return out

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(1))
    def forward(self, batch):
        return batch * F.sigmoid(self.beta * batch)

discriminator = nn.Sequential(
        nn.Conv2d(1, 4, 5),
        nn.LeakyReLU(0.2),
        nn.Conv2d(4, 8, 5),
        nn.LeakyReLU(0.2),
        nn.Conv2d(8, 16, 5),
        nn.LeakyReLU(0.2),
        Reshape(-1, 4096),
        nn.Linear(4096, 100),
        nn.LeakyReLU(0.2),
        nn.Linear(100, 1),
        )

transformer = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(32, 32, 3),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(32, 32, 3),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

encoder = nn.Sequential(
        nn.Conv2d(1, 4, 5),
        nn.LeakyReLU(0.2),
        nn.Conv2d(4, 8, 5),
        nn.LeakyReLU(0.2),
        nn.Conv2d(8, 16, 5),
        nn.LeakyReLU(0.2),
        Reshape(-1, 4096),
        nn.Linear(4096, 100),
        nn.LeakyReLU(0.2),
        nn.Linear(100, 2*latent_size),
        )

decoder = nn.Sequential(
        nn.Linear(latent_size, 100),
        nn.LeakyReLU(0.2),
        nn.Linear(100, 4096),
        nn.LeakyReLU(0.2),
        Reshape(-1, 16, 16, 16),
        nn.ConvTranspose2d(16, 8, 5),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(8, 4, 5),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(4, 1, 5),
        )


if __name__=="__main__":
    data_loader = get_data_loader(False)
    tot = 0
    for i in data_loader:
        img = i[0]
        pix = torch.sum(img[:, 0, 4, 4])
        tot += pix
    print(tot)
    # Insight, top left pixel always zero

    mean_images = get_mean_images()

    from visdom import Visdom
    vis = Visdom()
    vis.images(mean_images)
