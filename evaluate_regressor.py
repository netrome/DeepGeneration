# Usage: python evaluate_regressor.py --cuda --regressor ~/path/to/regressor/R.params  [--real-data --test | --generated ~/Data/DG1]
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import utils.utils as u
import utils.datasets as datasets
import utils.ellipse_detector as eld

import settings
import numpy as np
import json


R = u.create_regressor()
R.load_state_dict(torch.load(settings.REGRESSOR_PATH))


dataset = u.get_data_set()
if settings.GENERATED_PATH is not None:
    dataset = datasets.GeneratedWithMaps(settings.GENERATED_PATH)
    print("Using generated data set at: {}".format(settings.GENERATED_PATH))

data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=32,
                                          shuffle=True,
                                          pin_memory=True,
                                          drop_last=True)


def compute_euclidian_errors(targets, predictions):
    pred_map = predictions.data.round()
    target_map = targets.data

    errors = np.zeros(len(targets))
    for i in range(len(pred_map)):
        #p = pred_map[i].nonzero().float()
        #try:
        #    px, py = p[:, 1].mean(), p[:, 2].mean()
        #except IndexError:
        #    px, py = torch.ones(1)*127.5, torch.ones(1)*127.5

        #t = target_map[i].nonzero().float()
        #tx, ty = t[:, 1].mean(), t[:, 2].mean()
        # Compare mean with found ellipse
        px, py = eld.fit_ellipse(pred_map[i].cpu().numpy()[0])
        tx, ty = eld.fit_ellipse(target_map[i].cpu().numpy()[0])

        euclidian_error = ((px - tx) ** 2 + (py - ty) ** 2) ** 0.5
        errors[i] = float(euclidian_error)
    return errors

def jaccard_distance(targets, predictions):
    pred_map = predictions.data.round()
    target_map = targets.data

    errors = np.zeros(len(pred_map))
    for i in range(len(pred_map)):
        intersection = torch.sum((pred_map[i] + target_map[i]) > 1)
        union = torch.sum((pred_map[i] + target_map[i]) >= 1)
        errors[i] = float(1 - intersection/union)

    return errors


error_function = jaccard_distance  # Toggle error here


errors = np.zeros(len(dataset))
idx = 0
for i, batch in enumerate(data_loader):
    print("Processing batch {}/{}".format(i+1, len(data_loader)), end="\r")
    batch = Variable(batch)
    if settings.CUDA:
        batch = batch.cuda()

    patterns = batch[:, 0, :, :].unsqueeze(1)
    targets = batch[:, 1, :, :].unsqueeze(1)
    predictions = F.sigmoid(R(patterns))

    errs = error_function(targets, predictions)
    for err in errs:
        errors[idx] = err
        idx += 1

err = errors.mean()
print("Average error: {}".format(err))
json.dump(list(errors), open("evaluation_error.json", "w"))  # Save output for later analysis


