import torch
import torch.nn.functional as F
from torch.autograd import Variable

import utils.networks as nets
import utils.datasets as datasets

import settings


R = nets.ImageToImage()
R.load_state_dict(torch.load("working_model/R.params"))

if settings.CUDA:
    R.cuda()

dataset = datasets.SyntheticFullyAnnotated(settings.DATA_PATH)
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=32,
                                          shuffle=True,
                                          pin_memory=True,
                                          drop_last=True)


def compute_euclidian_error(targets, predictions):
    pred_map = predictions.data.round()
    target_map = targets.data

    sum_euclidian_error = 0
    for i in range(len(pred_map)):
        p = pred_map[i].nonzero().float()
        px, py = p[:, 1].mean(), p[:, 2].mean()

        t = target_map[i].nonzero().float()
        tx, ty = t[:, 1].mean(), t[:, 2].mean()

        euclidian_error = ((px - tx) ** 2 + (py - ty) ** 2) ** 0.5
        sum_euclidian_error += euclidian_error
    return sum_euclidian_error


err = 0
for i, batch in enumerate(data_loader):
    print("Processing batch {}/{}".format(i+1, len(data_loader)), end="\r")
    batch = Variable(batch)
    if settings.CUDA:
        batch = batch.cuda()

    patterns = batch[:, 0, :, :].unsqueeze(1)
    targets = batch[:, 1, :, :].unsqueeze(1)
    predictions = F.sigmoid(R(patterns))

    err += compute_euclidian_error(targets, predictions)

err /= len(dataset)
print("Average euclidian error: {}".format(err))

