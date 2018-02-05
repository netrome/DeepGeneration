import glob
import os
import json

import torch
import numpy as np
import cv2

from torch.utils.data import Dataset
import scipy.ndimage as image


def _load_image(path):
    pic = image.imread(path, mode="L")
    pic = pic.astype("float32") / 255  # Range [0, 1]
    w = pic.shape[0]
    pic = pic.reshape((1, w, w))  # Add third channel
    return pic


def _generate_heatmap(path, shape):
    meta = json.load(open(path, "r"))
    annotation = meta["pupil_optical_ellipse"]

    img = np.zeros([1, shape[2], shape[2]]).astype("float32")

    cv2.ellipse(img[0], (int(annotation[0]), int(annotation[1])),
                (int(annotation[2]), int(annotation[3])), int(annotation[4] * 180 / np.pi),
                0, 360, 255, -1)
    return img / 255


class SyntheticFullyAnnotated(Dataset):

    def __init__(self, data_root):
        self.image_paths = glob.glob(os.path.expanduser(
            os.path.join(data_root, "*.png")), recursive=False)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        pic = _load_image(self.image_paths[item])
        heatmap = _generate_heatmap(self.image_paths[item].strip(".png") + ".json", pic.shape)

        return torch.cat([torch.from_numpy(pic), torch.from_numpy(heatmap)])
