import numpy as np

import torch 
from torch.autograd import Variable
import torch.nn as nn

import iris
import networks

encoder = networks.encoder 

decoder = networks.decoder

# Minor data processing, concatenate the data sets
X, Y = iris.get_train_data()
data = np.concatenate([X, Y.reshape([75, 1])], axis=1)
complete_batch = Variable(torch.from_numpy(data).float())

opt = torch.optim.Adam([
    {"params": encoder.parameters()},
    {"params": decoder.parameters()},
    ])


iters = 20000
for i in range(iters):
    out = encoder(complete_batch)
    mu, log_var = out[:, :networks.latent_size], out[:, networks.latent_size:]

    # Compute loss
    std = log_var.mul(0.5).exp_()
    eps = Variable(std.data.new(std.size()).normal_())
    sampled = eps.mul(std).add_(mu).view(75, networks.latent_size)

    decoded = decoder(sampled)

    L1 = torch.mean(torch.abs(decoded - complete_batch))
    KLD =  -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / 75 / networks.latent_size

    loss = L1 + KLD * 0.1

    opt.zero_grad()
    loss.backward()
    opt.step()

    if i%100 == 0:
        print("{}/{}: loss: {}      ".format(i, iters, float(loss)), end="\r")
print()

torch.save(encoder.state_dict(), open("saved_nets/vae_encoder.params", "wb"))
torch.save(decoder.state_dict(), open("saved_nets/vae_decoder.params", "wb"))
print("Saved encoder and decoder")

