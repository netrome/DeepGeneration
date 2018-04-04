import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

import networks
import iris

import sys

decoder = networks.decoder
classifier = networks.classifier
latent = Variable(torch.FloatTensor(200, networks.latent_size).normal_())  # Latent codes to generate new data

# Load networks
decoder.load_state_dict(torch.load("saved_nets/{}_decoder.params".format(sys.argv[1])))

# Get original data
X, Y = iris.get_train_data()
X_tst, Y_tst = iris.get_test_data()
data = np.concatenate([X, Y.reshape([75, 1])], axis=1)
data_tst = np.concatenate([X_tst, Y_tst.reshape([75, 1])], axis=1)
complete_batch = Variable(torch.from_numpy(data).float())
test_batch = Variable(torch.from_numpy(data_tst).float())

# Generate new data
generated = decoder(latent)

# create train set
if "augment" in sys.argv:
    patterns = torch.cat((generated[:, :2], complete_batch[:, :2]), 0)
    targets = torch.cat((generated[:, 2], complete_batch[:, 2]), 0).view(275, 1)
elif "original" in sys.argv:
    patterns = complete_batch[:, :2]
    targets = complete_batch[:, 2].contiguous().view(75, 1)
elif "gen" in sys.argv:
    patterns = generated[:, :2]
    targets = generated[:, 2].contiguous().view(200, 1)

patterns = patterns.detach()
targets = targets.detach()

opt = torch.optim.Adam(classifier.parameters())

iters = 10000
for i in range(iters):
    pred = classifier(patterns)
    loss = torch.mean((pred - targets)**2)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if i % 100 == 0:
        print("{}/{}, loss: {}".format(i, iters, float(loss)), end="\r")
print()
print("------")
# Evaluate accuracy
test_pred = classifier(test_batch[:, :2])
mse = torch.mean((test_pred - test_batch[:, 2].contiguous().view(75, 1)) ** 2)
print("Mse: {}".format(float(mse)))

