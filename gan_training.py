""" GAN training scheme, carved out from aegan"""
import json
import time

import settings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

import utils.datasets as datasets
import utils.visualizer as vis
import utils.utils as u

from utils.utils import cyclic_data_iterator

G = u.create_generator()
D = u.create_discriminator()
toRGB = nn.Conv2d(16, 2, 1)
fromRGB = nn.Conv2d(2, 16, 1)  # Shared between discriminator and encoder
latent = Variable(torch.FloatTensor(settings.BATCH_SIZE, 128, 1, 1))
latent_ref_point = Variable(torch.FloatTensor(16, 128, 1, 1), volatile=True)
positive_targets = Variable(torch.ones(settings.BATCH_SIZE, 1))
negative_targets = Variable(torch.zeros(settings.BATCH_SIZE, 1))

pred_fake_history = Variable(torch.zeros(1), volatile=True)
pred_real_history = Variable(torch.zeros(1), volatile=True)

if settings.CUDA:
    toRGB.cuda()
    fromRGB.cuda()
    latent = latent.cuda()
    latent_ref_point = latent_ref_point.cuda()
    pred_fake_history = pred_fake_history.cuda()
    pred_real_history = pred_real_history.cuda()
    positive_targets = positive_targets.cuda()
    negative_targets = negative_targets.cuda()

opt_G = torch.optim.Adamax(G.parameters(), lr=settings.LEARNING_RATE, betas=settings.BETAS)
opt_D = torch.optim.Adamax(D.parameters(), lr=settings.LEARNING_RATE, betas=settings.BETAS)
opt_toRGB = torch.optim.Adamax(toRGB.parameters(), lr=settings.LEARNING_RATE, betas=settings.BETAS)
opt_fromRGB = torch.optim.Adamax(toRGB.parameters(), lr=settings.LEARNING_RATE, betas=settings.BETAS)

reconstruction_loss = nn.L1Loss()
#adversarial_loss = nn.MSELoss()

visualizer = vis.Visualizer()
state = json.load(open("working_model/state.json", "r"))
pred_real_history += state["pred_real"]
pred_fake_history += state["pred_fake"]
visualizer.point = state["point"]

if settings.WORKING_MODEL:
    print("Using model parameters in ./working_model")
    G.load_state_dict(torch.load("working_model/G.params"))
    D.load_state_dict(torch.load("working_model/D.params"))

    toRGB.load_state_dict(torch.load("working_model/toRGB6.params"))
    fromRGB.load_state_dict(torch.load("working_model/fromRGB6.params"))
    print("Loaded RGB layers too")

dataset = u.get_data_set()
data_loader = torch.utils.data.DataLoader(dataset,
                                          num_workers=8,
                                          batch_size=settings.BATCH_SIZE,
                                          shuffle=True,
                                          pin_memory=True,
                                          drop_last=True)


def update_visualization(visualizer, batch, fake, pred_fake, pred_real):
    # TODO move this to utils or visualizer to save code
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

    visualizer.update_loss(pred_real.data.cpu(), pred_fake.data.cpu())


def save_fake_reference_batch(point):
    if not settings.WORKING_MODEL:
        raise RuntimeError("Won't save reference batch without working model")
    torch.manual_seed(1337)
    latent_ref_point.data.normal_()
    torch.manual_seed(int(time.clock()*1e6))
    fake = toRGB(G(latent_ref_point))
    batch_shape = list(fake.shape)
    batch_shape[1] = 1
    single = make_grid(fake[:, 0].data.cpu().contiguous().view(batch_shape))
    save_image(single, "working_model/timelapse/fake_batch{}.png".format(point))

update_state = 0
for chunk in range(settings.CHUNKS):
    print("Chunk {}/{}    ".format(chunk, settings.CHUNKS))
    batch, fake = None, None
    for i, batch in enumerate(cyclic_data_iterator(data_loader, settings.STEPS)):
        batch = Variable(batch)
        if settings.CUDA:
            batch = batch.cuda()
        latent.data.normal_()  # Sample latent vector

        fake = toRGB(G(latent))
        pred_fake = D(fromRGB(fake))

        if update_state == settings.DISCRIMINATOR_ITERATIONS:
            update_state = 0

            adv_loss = - torch.mean(pred_fake)
            loss = adv_loss 

            # Perform an optimization step
            opt_G.zero_grad()
            opt_toRGB.zero_grad()
            loss.backward()
            opt_G.step()
            opt_toRGB.step()
        else:
            update_state += 1
            pred_real = D(fromRGB(batch))
            pred_real_history = pred_real_history * 0.9 + torch.mean(pred_real) * 0.1
            pred_fake_history = pred_fake_history * 0.9 + torch.mean(pred_fake) * 0.1
            loss = torch.mean(pred_fake - pred_real)  # Wasserstein loss

            # Add gradient penalty
            grads = torch.autograd.grad(torch.mean(pred_fake), D.parameters(),
                                        create_graph=True, allow_unused=True)
            grad_norm = 0
            for grad in grads:
                if grad is not None:
                    grad_norm += grad.pow(2).sum()
            grad_norm.sqrt_()

            grad_loss = (grad_norm - 1).pow(2)
            loss += 10 * grad_loss
            loss += 0.0001 * torch.mean(pred_real).pow(2)  # Drift loss

            # Perform an optimization step
            opt_fromRGB.zero_grad()
            opt_D.zero_grad()
            loss.backward()
            opt_D.step()
            opt_fromRGB.step()

        if i % 10 == 9:
            print("Step {}/{}   ".format(i + 1, settings.STEPS), end="\r")

    state["history_real"].append(float(pred_real_history))
    state["history_fake"].append(float(pred_fake_history))
    update_visualization(visualizer, batch, fake, pred_fake_history, pred_real_history)
    save_fake_reference_batch(visualizer.point)

# Save models
print("Saving rgb layers, {}".format(time.ctime()))

torch.save(toRGB.state_dict(), "working_model/toRGB6.params")
torch.save(fromRGB.state_dict(), "working_model/fromRGB6.params")
torch.save(G.state_dict(), "working_model/G.params")
torch.save(D.state_dict(), "working_model/D.params")

# Save state
state["point"] = visualizer.point
state["pred_real"] = float(pred_real_history)
state["pred_fake"] = float(pred_fake_history)
print("Saving state, {}".format(time.ctime()))
json.dump(state, open("working_model/state.json", "w"))

print("Finished with main")