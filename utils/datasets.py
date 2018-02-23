import glob
import os
import json
import random

import torch
import torch.nn.functional as F
import numpy as np
import cv2

from torch.utils.data import Dataset
import scipy.ndimage as image


def _load_image(path, dim=None):
    pic = image.imread(path, mode="L")
    pic = pic.astype("float32") / 255  # Range [0, 1]

    if dim is None:
        w = pic.shape[0]
        h = w
    else:
        w, h = dim
    pic = pic.reshape((1, w, h))  # Add third channel
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


class GeneratedWithMaps(Dataset):
    def __init__(self, data_root):
        self.image_paths = glob.glob(os.path.expanduser(
            os.path.join(data_root, "image*.png")), recursive=False)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        pic = _load_image(self.image_paths[item])
        heatmap = _load_image(self.image_paths[item].replace("image", "map"))

        return torch.cat([torch.from_numpy(pic), torch.from_numpy(heatmap)])


class DeepGazeData(Dataset):
    def __init__(self, test=False):
        name = "test" if test else "train"
        self.meta_files = json.load(open("lists/DG_{}.json".format(name), "r"))

    def __len__(self):
        return len(self.meta_files)

    def __getitem__(self, item):
        file = self.meta_files[item]

        meta = json.load(open(file, "r"))
        region, tracks = meta["cropped_region"], meta["markup"]["tracks"]
        pupils = [i for i in tracks if i["name"] == "pupil"]
        pupil = random.choice(pupils)
        x, y = pupil["points"][0]["x"], pupil["points"][0]["y"]
        wx, wy = pupil["radius"]["x"], pupil["radius"]["y"]
        rotation = pupil["rotation_angle"]

        w, h = region["width"], region["height"]

        # Generate heatmap
        x, y, wx, wy = [int(round(i)) for i in (x, y, wx, wy)]
        heatmap = np.zeros([1, h, w]).astype("float32")
        cv2.ellipse(heatmap[0], (x, y), (wx, wy), rotation, 0, 360, 255, -1)

        pic = _load_image(file.replace(".json", ".png"), (h, w))
        tot = torch.cat([torch.from_numpy(pic), torch.from_numpy(heatmap)/255])

        # Crop image
        min_x = max(0, x - 200)
        max_x = min(w - 256, x - 56)
        max_y = h - 256
        start = (random.randint(0, max_y), random.randint(min_x, max_x))

        return tot[:, start[0]:start[0] + 256, start[1]:start[1] + 256]

if __name__ == "__main__":
    data = DeepGazeData()
    import code
    code.interact(local=locals())
