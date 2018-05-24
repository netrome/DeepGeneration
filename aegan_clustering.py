import os

import numpy as np
import torch
from torch.autograd import Variable
import settings
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import utils.utils as u

fromRGB = torch.nn.Conv2d(2, 16, 1)
encoder = u.create_encoder()
encoder.load_state_dict(torch.load(os.path.join(settings.MODEL_PATH, "E.params")))
fromRGB.load_state_dict(torch.load(os.path.join(settings.MODEL_PATH, "fromRGB6.params")))

data = u.get_data_set()
data_loader = torch.utils.data.DataLoader(data, batch_size=30)

latent = torch.zeros(len(data), 128)

if settings.CUDA:
    encoder.cuda()
    fromRGB.cuda()

# Transform data set to latent codes

print("Doing stuff")
for i, img in enumerate(data_loader):
    if settings.CUDA:
        img = img.cuda()
    latent[30*i:30*(i+1)] = encoder(fromRGB(Variable(img)))[0].data
    print(i)
print("Done stuff")

# Reduce dimensionality with PCA

latent = latent.numpy()

pca = PCA(n_components=2)
pca.fit(latent)
print("PCA learned")

X = pca.transform(latent)

# Plot transformed data
plt.scatter(X[:, 0], X[:, 1])
plt.show()
