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

discriminator = networks.classifier

latent = Variable(torch.FloatTensor(75, networks.latent_size).normal_())

# Minor data processing, concatenate the data sets
X, Y = iris.get_train_data()
data = np.concatenate([X, Y.reshape([75, 1])], axis=1)
complete_batch = Variable(torch.from_numpy(data).float())

opt_GE = torch.optim.Adam([
    {"params": encoder.parameters()},
    {"params": decoder.parameters()},
    ])

opt_discriminator = torch.optim.Adam([
    {"params": discriminator.parameters()},
    ])

update_state = 0
iters = 20000
for i in range(iters):
    latent.data.normal_()
    generated = decoder(latent)

    pred_real = discriminator(complete_batch)
    pred_fake = discriminator(generated)

    if update_state == 10:  # Number of discriminator iterations
        update_state = 0

        loss = -torch.mean(pred_fake)

        opt_GE.zero_grad()
        loss.backward()
        opt_GE.step()
    else:
        update_state += 1
        loss = torch.mean(pred_fake) - torch.mean(pred_real)
        
        gradients = torch.autograd.grad(torch.mean(pred_real), discriminator.parameters(),
                                                    create_graph=True, allow_unused=True)
        grad_norm = 0
        for grad in gradients:
            if grad is not None:  # This indicates something weird
                grad_norm += grad.pow(2).sum()
        grad_norm = grad_norm.sqrt()
        grad_loss = ((grad_norm - 1).pow(2))

        loss += 10 * grad_loss

        opt_discriminator.zero_grad()
        loss.backward()
        opt_discriminator.step()

    if i%100 == 0:
        print("{}/{}: loss: {}      ".format(i, iters, float(loss)), end="\r")
print()

torch.save(encoder.state_dict(), open("saved_nets/wgan_encoder.params", "wb"))
torch.save(decoder.state_dict(), open("saved_nets/wgan_decoder.params", "wb"))
print("Saved encoder and decoder")

