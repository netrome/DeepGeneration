import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

import networks
import iris

import sys

encoder = networks.encoder
decoder = networks.decoder
latent = Variable(torch.FloatTensor(200, networks.latent_size).normal_(), volatile=True)  # Latent codes to generate new data

# Load networks
encoder.load_state_dict(torch.load("saved_nets/{}_encoder.params".format(sys.argv[1])))
decoder.load_state_dict(torch.load("saved_nets/{}_decoder.params".format(sys.argv[1])))

# Get original data
X, Y = iris.get_train_data()
data = np.concatenate([X, Y.reshape([75, 3])], axis=1)
complete_batch = Variable(torch.from_numpy(data).float(), volatile=True)

# Encode-decode data
encoded = encoder(complete_batch)[:, :networks.latent_size].contiguous().view(75, networks.latent_size)
decoded = decoder(encoded)

# Generate new data
generated = decoder(latent)


# Convert generated data to numpy
decoded = decoded.data.numpy()
generated = generated.data.numpy()

plt.scatter(X[:, 0], X[:, 1], c=iris.ohe_to_idx(Y))
plt.figure()
plt.scatter(decoded[:, 0], decoded[:, 1], c=iris.ohe_to_idx(decoded[:, 2:]))
plt.figure()
plt.scatter(generated[:, 0], generated[:, 1], c=iris.ohe_to_idx(generated[:, 2:]))
plt.show()

