import torch
from torch.autograd import Variable
import utils as u
from visdom import Visdom

import sys

latent = Variable(torch.FloatTensor(256, u.latent_size))  # Generate and visualize, don't save
latent.data.normal_()
G = u.decoder 

G.load_state_dict(torch.load(open(sys.argv[1], "rb")))

fake = G(latent)

vis = Visdom()
vis.images(fake.data.clamp(0, 1))

