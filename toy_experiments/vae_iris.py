import numpy as np

import torch 
from torch.autograd import Variable
import torch.nn as nn

import iris

encoder = nn.Sequential(
        nn.Linear(3, 10),
        nn.ReLU(),
        nn.Linear(10, 2)
        )

decoder = nn.Sequential(
        nn.Linear(1, 10),
        nn.ReLU(),
        nn.Linear(10, 3)
        )

# Minor data processing, concatenate the data sets
X, Y = iris.get_train_data()
data = np.concatenate([X, Y.reshape([75, 1])], axis=1)
complete_batch = Variable(torch.from_numpy(data).float())

opt = torch.optim.Adam([
    {"params": encoder.parameters()},
    {"params": decoder.parameters()},
    ])


for i in range(10000):
    out = encoder(complete_batch)
    mu, log_var = out[:, 0], out[:, 1]

    # Compute loss
    std = log_var.mul(0.5).exp_()
    eps = Variable(std.data.new(std.size()).normal_())
    sampled = eps.mul(std).add_(mu).view(75, 1)

    decoded = decoder(sampled)

    MSE = torch.mean((decoded - complete_batch) ** 2)
    KLD =  -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / 75

    loss = MSE + KLD
    print(float(loss))

    opt.zero_grad()
    loss.backward()
    opt.step()

torch.save(encoder.state_dict(), open("saved_nets/vae_encoder.params", "wb"))
torch.save(decoder.state_dict(), open("saved_nets/vae_decoder.params", "wb"))
print("Saved encoder and decoder")

