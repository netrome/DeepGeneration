import torch
from torch.autograd import Variable
import utils as u
from visdom import Visdom

import sys

data_loader = u.get_data_loader()
#mean_images = u.get_mean_images()
E = u.encoder 
G = u.decoder 

vis = Visdom()

if "cuda" in sys.argv:
    E = E.cuda()
    G = G.cuda()

opt = torch.optim.Adam([
    {'params': E.parameters()},
    {'params': G.parameters()},
        ])

ref = torch.arange(0, 64).long()

print("Ready to train")
for epoch in range(10):
    for i, (img, label) in enumerate(data_loader):
        #img[:, 0, 0, :][ref, label*2] = 1
        #img[:, 0, 0, :][ref, label*2+1] = 1
        #img[:, 0, 1, :][ref, label*2] = 1
        #img[:, 0, 1, :][ref, label*2+1] = 1
        #img[:, 0, 2, :][ref, label*2] = 1
        #img[:, 0, 2, :][ref, label*2+1] = 1
        cat = Variable(img)
        #cat = torch.cat([img, mean_images[label]], dim=1)
        #cat = Variable(cat)

        if "cuda" in sys.argv:
            cat = cat.cuda()

        out = E(cat)
        mu, log_var = out[:, :u.latent_size], out[:, u.latent_size:]

        # Reparametrization-sampling
        std = log_var.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        sampled = eps.mul(std).add_(mu).view(64, u.latent_size)

        decoded = G(sampled)

        L1 = torch.mean(torch.abs(decoded - cat))
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / 64 / u.latent_size

        loss = L1 + KLD * 0.01

        opt.zero_grad()
        loss.backward()
        opt.step()
        if i%50 == 0:
            print("Iter {}, loss: {}   ".format(i, float(loss)), end="\r")
    vis.images(cat[:, 0, :, :].data.cpu().view(64, 1, 28, 28), win="original")
    vis.images(decoded[:, 0, :, :].data.cpu().clamp(0, 1).view(64, 1, 28, 28), win="fake")
    #vis.images(decoded[:, 1, :, :].data.cpu().clamp(0, 1).view(64, 1, 28, 28), win="labels")
    print("Epoch {}/10, loss: {}".format(epoch, float(loss)))

torch.save(E.state_dict(), open("saved_nets/vae_encoder.params", "wb"))
torch.save(G.state_dict(), open("saved_nets/vae_decoder.params", "wb"))

