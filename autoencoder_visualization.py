import torch
import torch.nn as nn

from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

import utils.utils as u
import utils.datasets as datasets

import settings
import os

E = u.create_encoder()
G = u.create_generator()

toRGB = nn.Conv2d(16, 2, 1)
fromRGB = nn.Conv2d(2, 16, 1)

dataset = u.get_data_set()
torch.random.manual_seed(1337)
data_loader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=8, shuffle=True)

G.load_state_dict(torch.load(os.path.join(settings.MODEL_PATH, "G.params")))
E.load_state_dict(torch.load(os.path.join(settings.MODEL_PATH, "E.params")))
toRGB.load_state_dict(torch.load(os.path.join(settings.MODEL_PATH, "toRGB6.params")))
fromRGB.load_state_dict(torch.load(os.path.join(settings.MODEL_PATH, "fromRGB6.params")))

if settings.CUDA:
    toRGB.cuda()
    fromRGB.cuda()

for batch in data_loader:
    batch = Variable(batch)
    if settings.CUDA:
        batch = batch.cuda()

    encoded = E(fromRGB(batch))[0]
    decoded = toRGB(G(encoded.view(-1, 128, 1, 1)))

    original = make_grid(batch[:, 0].data.cpu().contiguous().view(8, 1, 256, 256))
    single = make_grid(decoded[:, 0].data.cpu().contiguous().view(8, 1, 256, 256))
    save_image(original, "/tmp/original.png")
    save_image(single, "/tmp/decoded.png")
    break

print("Done")
