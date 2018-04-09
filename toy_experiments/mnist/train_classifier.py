import torch
import torch.nn as nn
from torch.autograd import Variable
import utils as u
from visdom import Visdom

import sys

C = u.classifier
data_loader = u.get_data_loader()

opt = torch.optim.Adam(C.parameters())
criterion = nn.BCEWithLogitsLoss()
ref = torch.arange(0, 64).long()
one_hot = Variable(torch.zeros(64, 10))

if "cuda" in sys.argv:
    ref = ref.cuda()
    one_hot = one_hot.cuda()
    C.cuda()

epochs = 10
for epoch in range(epochs):
    for i, (img, label) in enumerate(data_loader):
        if "cuda" in sys.argv:
            img = img.cuda()
            label = label.cuda()
        one_hot[ref, label] = 1
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
