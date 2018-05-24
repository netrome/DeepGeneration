import torch
import torch.nn.functional as F
from torch.autograd import Variable

import utils.utils as u
import utils.datasets as datasets
import utils.ellipse_detector as eld

import settings
import numpy as np
import json

from torchvision.utils import save_image

R = u.create_regressor()
R.load_state_dict(torch.load(settings.REGRESSOR_PATH))


dataset = u.get_data_set()
if settings.GENERATED_PATH is not None:
    dataset = datasets.GeneratedWithMaps(settings.GENERATED_PATH)
    print("Using generated data set at: {}".format(settings.GENERATED_PATH))

img = dataset[5][0].view(1, 256, 256)
print(img.shape)

pred = R(Variable(img.view(1, 1, 256, 256)))[0]
print(pred.shape)

# Save image and map in /tmp/
save_image(img, "/tmp/ref.png")
save_image(pred.data, "/tmp/pred.png")
