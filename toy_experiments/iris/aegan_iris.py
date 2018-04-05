import numpy as np

import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import iris
import networks

import sys

encoder = networks.encoder 

decoder = networks.decoder

discriminator = networks.discriminator

latent = Variable(torch.FloatTensor(75, networks.latent_size).normal_())

# Minor data processing, concatenate the data sets
X, Y = iris.get_train_data()
data = np.concatenate([X, Y.reshape([75, 1])], axis=1)
complete_batch = Variable(torch.from_numpy(data).float())

opt_GE = torch.optim.Adam([
    {"params": encoder.parameters()},
    {"params": decoder.parameters()},
    ])

opt_D = torch.optim.Adam([
    {"params": discriminator.parameters()},
    ])

update_state = 0
iters = 20000
for i in range(iters):
    latent.data.normal_()
    out = encoder(complete_batch)
    encoded = out[:, :networks.latent_size]
    decoded = decoder(encoded)
    generated = decoder(latent)

    pred_real = discriminator(complete_batch)
    pred_fake = discriminator(generated)

    if update_state == 1:  # Number of discriminator iterations
        update_state = 0

        L1 = torch.mean(torch.abs(decoded - complete_batch))
        adversarial_loss = torch.mean((pred_fake - 1).pow(2))
        drift_loss = torch.mean(F.relu(encoded.norm(2, 1) - 1))

        loss = L1 + adversarial_loss + drift_loss

        opt_GE.zero_grad()
        loss.backward()
        opt_GE.step()
    else:
        update_state += 1
        loss = torch.mean((pred_real - 1)**2 + pred_fake ** 2)

        opt_D.zero_grad()
        loss.backward()
        opt_D.step()

    if i%100 == 0:
        print("{}/{}: loss: {}      ".format(i, iters, float(loss)), end="\r")
print()

torch.save(encoder.state_dict(), open("saved_nets/aegan_encoder.params", "wb"))
torch.save(decoder.state_dict(), open("saved_nets/aegan_decoder.params", "wb"))
print("Saved encoder and decoder")

