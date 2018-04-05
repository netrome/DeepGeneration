import torch
import torch.nn.functional as F
from torch.autograd import Variable
import utils as u
from visdom import Visdom

import sys

data_loader = u.get_data_loader()
mean_images = u.get_mean_images()
latent = Variable(torch.FloatTensor(64, 20))
E = u.encoder 
G = u.decoder 
D = u.discriminator

vis = Visdom()

if "cuda" in sys.argv:
    E.cuda()
    G.cuda()
    D.cuda()
    latent = latent.cuda()

opt = torch.optim.Adamax([
    {'params': E.parameters()},
    {'params': G.parameters()},
        ])

opt_D = torch.optim.Adamax([
    {'params': D.parameters()},
    ], lr=0.0002, betas=(0.5, 0.99))

print("Ready to train")
update_state = 0
epochs = 100
for epoch in range(epochs):
    for i, (img, label) in enumerate(data_loader):
        latent.data.normal_()
        cat = torch.cat([img, mean_images[label]], dim=1)
        cat = Variable(cat)

        if "cuda" in sys.argv:
            cat = cat.cuda()

        fake = G(latent)
        
        pred_real = D(cat)
        pred_fake = D(fake)

        if update_state == 3:
            update_state = 0

            loss = -torch.mean(pred_fake)

            opt.zero_grad()
            loss.backward()
            opt.step()
        else:
            update_state += 1
            loss = torch.mean(pred_fake) - torch.mean(pred_real)

            gradients = torch.autograd.grad(torch.mean(pred_real), D.parameters(), 
                    create_graph=True, allow_unused=True)

            grad_norm = 0
            for grad in gradients:
                if grad is not None:
                    grad_norm += grad.pow(2).sum()
            grad_norm = grad_norm.sqrt()
            grad_loss = ((grad_norm - 1).pow(2))

            loss += 10*grad_loss

            opt_D.zero_grad()
            loss.backward()
            opt_D.step()

        if i%50 == 0:
            print("Iter {}, loss: {}                   ".format(i, float(loss)), end="\r")

    vis.images(cat[:, 0, :, :].data.cpu().view(64, 1, 28, 28), win="original")
    vis.images(fake[:, 0, :, :].data.cpu().clamp(0, 1).view(64, 1, 28, 28), win="fake1", opts={"caption": "fake images"})
    vis.images(fake[:, 1, :, :].data.cpu().clamp(0, 1).view(64, 1, 28, 28), win="label1", opts={"caption": "fake labels"})
    print("Epoch {}/{}, loss: {}".format(epoch, epochs, float(loss)))

torch.save(E.state_dict(), open("saved_nets/wgan_encoder.params", "wb"))
torch.save(G.state_dict(), open("saved_nets/wgan_decoder.params", "wb"))

