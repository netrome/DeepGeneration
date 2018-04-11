import torch
import torch.nn as nn
from torch.autograd import Variable
import utils as u
from visdom import Visdom

import sys

C = u.classifier
C.load_state_dict(torch.load(open(sys.argv[1], "rb")))
data_loader = u.get_data_loader(train=False, batch_size=100)
data_loader.drop_last = False

opt = torch.optim.Adam(C.parameters())
ref = torch.arange(0, 100).long()
one_hot = Variable(torch.zeros(100, 10))

if "cuda" in sys.argv:
    ref = ref.cuda()
    one_hot = one_hot.cuda()
    C.cuda()

epochs = 10
corrects = 0
total = 0
for epoch in range(1):
    for i, (img, label) in enumerate(data_loader):
        if "cuda" in sys.argv:
            img = img.cuda()
            label = label.cuda()
        one_hot[ref, label] = 1
        pred = C(Variable(img, volatile=True))
        pred[ref, pred.max(1)[1]] = 1
        pred[pred != 1] = 0
        corrects += float(torch.sum(pred.round() * one_hot))
        total += len(label)

        one_hot[ref, label] = 0

        if i % 30 == 0:
            print("Iter: {}/{}    ".format(i, len(data_loader)), end="\r")

print("Results ---------------")
print("Corrects: {}".format(corrects))
print("Errs: {}".format(total - corrects))
print("Total: {}".format(total))
print("Accuracy: {}".format(corrects/total))

