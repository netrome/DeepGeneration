import torch
import torch.nn as nn
from torch.autograd import Variable
import utils as u
from visdom import Visdom

import sys

C = u.classifier
data_loader = u.get_data_loader()

if "augment" in sys.argv:
    print("Using augmented data")
    E = u.encoder
    G = u.decoder 

    E.load_state_dict(torch.load(open("saved_nets/{}_encoder.params".format(sys.argv[1]), "rb")))
    G.load_state_dict(torch.load(open("saved_nets/{}_decoder.params".format(sys.argv[1]), "rb")))

opt = torch.optim.Adam(C.parameters())
criterion = nn.BCEWithLogitsLoss()
ref = torch.arange(0, 64).long()
one_hot = Variable(torch.zeros(64, 10))

if "cuda" in sys.argv:
    ref = ref.cuda()
    one_hot = one_hot.cuda()
    C.cuda()
    E.cuda()
    G.cuda()

epochs = 10
for epoch in range(epochs):
    for i, (img, label) in enumerate(data_loader):
        if "cuda" in sys.argv:
            img = img.cuda()
            label = label.cuda()
        one_hot[ref, label] = 1

        if "augment" in sys.argv:
            out = E(Variable(img))
            mu, log_var = out[:, :u.latent_size], out[:, u.latent_size:]
            std = log_var.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            sampled = eps.mul(std).add_(mu).view(64, u.latent_size)
            fake = G(sampled)
            fake = fake.detach()
            
            pred = C(fake)
            loss = criterion(pred, one_hot)

            opt.zero_grad()
            loss.backward()
            opt.step()

        pred = C(Variable(img))
        loss = criterion(pred, one_hot)

        opt.zero_grad()
        loss.backward()
        opt.step()

        one_hot[ref, label] = 0

        if i % 100 == 0:
            print("Iter: {}/{}, loss: {}    ".format(i, len(data_loader), float(loss)), end="\r")
    print("Epoch {}/{}, loss {}           ".format(epoch, epochs, float(loss)))

# Save classifier
torch.save(C.state_dict(), open("classifier.params", "wb"))
