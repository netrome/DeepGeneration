import os

import numpy as np
import torch
from torch.autograd import Variable
import settings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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

#pca = PCA(n_components=2)
tsne = TSNE(n_components=2)
#pca.fit(latent)
X = tsne.fit_transform(latent)
print("TSNE learned and used")

# Plot transformed data
fig2, ax2 = plt.subplots()
def onclick(event):
    point = event.artist
    ind = event.ind 
    print("Halloj: {}".format(ind))

    for i in ind:
        img = data[i][0,:,:]
        ax2.imshow(img, cmap="gray")
        fig2.show()

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], picker=5)
fig.canvas.mpl_connect("pick_event", onclick)
plt.show()
