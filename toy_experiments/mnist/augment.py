import torch
from torch.autograd import Variable
import utils as u
from visdom import Visdom

import sys

data_loader = u.get_data_loader(batch_size=256)

E = u.encoder
G = u.decoder 

E.load_state_dict(torch.load(open("saved_nets/{}_encoder.params".format(sys.argv[1]), "rb")))
G.load_state_dict(torch.load(open("saved_nets/{}_decoder.params".format(sys.argv[1]), "rb")))


for batch, labels in data_loader:
    out = E(Variable(batch, volatile=True))
    mu, log_var = out[:, :u.latent_size], out[:, u.latent_size:]
    std = log_var.mul(0.5).exp_()
    eps = Variable(std.data.new(std.size()).normal_())
    sampled = eps.mul(std).add_(mu).view(256, u.latent_size)
    fake = G(sampled)
    break


vis = Visdom()
vis.images(fake.data.clamp(0, 1))

