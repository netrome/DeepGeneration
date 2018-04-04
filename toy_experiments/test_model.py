import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

import networks
import iris

encoder = networks.encoder()
decoder = networks.decoder()
latent = torch.FloatTensor(200, 1).random_()  # Latent codes to generate new data

# Load networks
encoder.load_state_dict(torch.load("saved_nets/vae_encoder.params"))
decoder.load_state_dict(torch.load("saved_nets/vae_decoder.params"))

# Get original data
X, Y = iris.get_train_data()
data = np.concatenate([X, Y.reshape([75, 1])], axis=1)
complete_batch = Variable(torch.from_numpy(data).float())

# Encode-decode data
encoded = encoder(Variable(complete_batch, volatile=True))[:, 0]
decoded = decoder(encoded)

# Generate new data
generated = decoder(latent)


# Convert generated data to numpy
decoded = decoded.data.numpy()
generated = generated.dara.numpy()

plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.figure()
plt.scatter(decoded[:, 0], decoded[:, 1], c=decoded[:, 2])
plt.show()

