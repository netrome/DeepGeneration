""" Pure VAE training, good for testing different architectures and stuff """
import json
import time
import math

import settings
import torch
import torch.nn as nn
from torch.autograd import Variable

import utils.networks as nets
import utils.progressive_networks as pnets
import utils.datasets as datasets
import utils.visualizer as vis

from utils.utils import cyclic_data_iterator

encoder = nets.TrivialEncoderLight()
decoder = pnets.TrivialGeneratorLight()
toRGB = nn.Conv2d(16, 2, 1)
fromRGB = nn.Conv2d(2, 16, 1)

if settings.CUDA:
    encoder.cuda()
    decoder.cuda()
    toRGB.cuda()
    fromRGB.cuda()

optimizer = torch.optim.Adamax([
    {"params": encoder.parameters()},
    {"params": decoder.parameters()},
    {"params": toRGB.parameters()},
    {"params": fromRGB.parameters()},
])

reconstruction_loss = nn.MSELoss()
KL_weight = 1e-6

visualizer = vis.Visualizer()
state = json.load(open("working_model/state.json", "r"))
visualizer.point = state["point"]

if settings.WORKING_MODEL:
    print("Using model parameters in ./working_model")
    decoder.load_state_dict(torch.load("working_model/G.params"))
    encoder.load_state_dict(torch.load("working_model/E.params"))

    toRGB.load_state_dict(torch.load("working_model/toRGB6.params"))
    fromRGB.load_state_dict(torch.load("working_model/fromRGB6.params"))
    print("Loaded RGB layers too")

dataset = datasets.SyntheticFullyAnnotated(settings.DATA_PATH)
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=settings.BATCH_SIZE,
                                          shuffle=True,
                                          pin_memory=True,
                                          drop_last=True)


def sample_with_reparametrization(mu, log_var):
    std = log_var.mul(0.1).exp_()
    eps = Variable(std.data.new(std.size()).normal_())
    return eps.mul(std).add_(mu)


def VAE_loss(decoded, original, mu, log_var):
    recon = reconstruction_loss(decoded, original)

    KLD_loss = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon, KLD_loss


def update_visualization(visualizer, batch, fake, MSE, KLD):
    batch_shape = list(batch.shape)
    batch_shape[1] = 1

    visualizer.update_image(batch[0][0].data.cpu(), "real_img")
    visualizer.update_image(batch[0][1].data.cpu(), "real_map")
    visualizer.update_image(batch[0].mean(0).data.cpu(), "real_cat")
    visualizer.update_batch(
        batch[:, 0].data.cpu().contiguous().view(batch_shape), "real_batch")

    fake.data.clamp_(0, 1)
    visualizer.update_image(fake[0][0].data.cpu(), "fake_img")
    visualizer.update_image(fake[0][1].data.cpu(), "fake_map")
    visualizer.update_image(fake[0].mean(0).data.cpu(), "fake_cat")

    visualizer.update_batch(
        fake[:, 0].data.cpu().contiguous().view(batch_shape), "fake_batch")

    visualizer.update_loss(MSE.data.cpu(), KLD.data.cpu())


for chunk in range(settings.CHUNKS):
    print("Chunk {}/{}    ".format(chunk, settings.CHUNKS))
    batch, decoded, MSE, KLD = None, None, None, None
    for i, batch in enumerate(cyclic_data_iterator(data_loader, settings.STEPS)):
        batch = Variable(batch)
        if settings.CUDA:
            batch = batch.cuda()

        encoded = encoder(fromRGB(batch))
        sampled = sample_with_reparametrization(encoded[0], encoded[1])
        decoded = toRGB(decoder(sampled.view(-1, 128, 1, 1)))  # No sigmoid

        recon, KLD = VAE_loss(decoded, batch, encoded[0], encoded[1])
        loss = recon + KLD * KL_weight

        # Perform an optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 9:
            print("Step {}/{}   ".format(i + 1, settings.STEPS), end="\r")

    update_visualization(visualizer, batch, decoded, recon, KLD)

# Save models
print("Saving rgb layers, {}".format(time.ctime()))

torch.save(toRGB.state_dict(), "working_model/toRGB6.params")
torch.save(fromRGB.state_dict(), "working_model/fromRGB6.params")
torch.save(decoder.state_dict(), "working_model/G.params")
torch.save(encoder.state_dict(), "working_model/E.params")

# Save state
state["point"] = visualizer.point
print("Saving state, {}".format(time.ctime()))
json.dump(state, open("working_model/state.json", "w"))

print("Finished with main")

