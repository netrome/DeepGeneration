import torch
import torch.nn.functional as F
from torch.autograd import Variable
import utils as u
from visdom import Visdom

import sys

data_loader = u.get_data_loader()
#mean_images = u.get_mean_images()
latent = Variable(torch.FloatTensor(64, u.latent_size))
E = u.encoder 
G = u.decoder 
D = u.discriminator

vis = Visdom()

if "cuda" in sys.argv:
    E.cuda()
    G.cuda()
    D.cuda()
    latent = latent.cuda()

opt = torch.optim.Adam([
    {'params': E.parameters()},
    {'params': G.parameters()},
    ])

opt_D = torch.optim.Adamax([
    {'params': D.parameters()},
    ], betas=(0.5, 0.99))

print("Ready to train")
ref = torch.arange(0, 64).long()
update_state = 0
epochs = 100
for epoch in range(epochs):
    for i, (img, label) in enumerate(data_loader):
        #img[:, 0, 0, :][ref, label*2] = 1
        #img[:, 0, 0, :][ref, label*2+1] = 1
        #img[:, 0, 1, :][ref, label*2] = 1
        #img[:, 0, 1, :][ref, label*2+1] = 1
        #img[:, 0, 2, :][ref, label*2] = 1
        #img[:, 0, 2, :][ref, label*2+1] = 1
        cat = Variable(img)
        #cat = torch.cat([img, mean_images[label]], dim=1)
        #cat = Variable(cat)
        latent.data.normal_()

        if "cuda" in sys.argv:
            cat = cat.cuda()

        out = E(cat)
        encoded = out[:, :u.latent_size]
        decoded = G(encoded)
        fake = G(latent)
        
        pred_real = D(cat)
        pred_fake = D(fake)

        if update_state == 3:  # Discriminator iterations
            update_state = 0

            L1 = torch.mean(torch.abs(decoded - cat)) 
            adv_loss = torch.mean((pred_fake - 1).pow(2))
            drift_loss = torch.mean(F.relu(encoded.norm(2, 1) - 1))

            loss = L1 + drift_loss + adv_loss * 1e-2

            opt.zero_grad()
            loss.backward()
            opt.step()
        else:
            update_state += 1
            loss = torch.mean((pred_real - 1) ** 2 + pred_fake ** 2) 

            opt_D.zero_grad()
            loss.backward()
            opt_D.step()

        if i%50 == 0:
            print("Iter {}, loss: {}                   ".format(i, float(loss)), end="\r")

    vis.images(cat[:, 0, :, :].data.cpu().view(64, 1, 28, 28), win="original")
    vis.images(decoded[:, 0, :, :].data.cpu().clamp(0, 1).view(64, 1, 28, 28), win="fake")
    vis.images(fake[:, 0, :, :].data.cpu().clamp(0, 1).view(64, 1, 28, 28), win="fake1", opts={"caption": "fake images"})
    print("Epoch {}/{}, loss: {}".format(epoch, epochs, float(loss)))

torch.save(E.state_dict(), open("saved_nets/aegan_encoder.params", "wb"))
torch.save(G.state_dict(), open("saved_nets/aegan_decoder.params", "wb"))

