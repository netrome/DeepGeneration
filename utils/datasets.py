import glob
import os
import json
import random
import csv

import torch
import torch.nn.functional as F
import numpy as np
import cv2

from torch.utils.data import Dataset
import scipy.ndimage as image
from torch.autograd import Variable


def _load_image(path, dim=None):
    pic = image.imread(path, mode="L")
    pic = pic.astype("float32") / 255  # Range [0, 1]

    if dim is None:
        w = pic.shape[0]
        h = pic.shape[1]
    else:
        w, h = dim
    pic = pic.reshape((1, w, h))  # Add third channel
    return pic


def _generate_heatmap(path, shape):
    meta = json.load(open(path, "r"))
    annotation = meta["pupil_optical_ellipse"]

    img = np.zeros([1, shape[2], shape[2]]).astype("float32")

    cv2.ellipse(img[0], (int(annotation[0]), int(annotation[1])),  # This should be rounded correctly
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


class HelenData(Dataset):
    def __init__(self, test=False):
        self.helen_dir = os.path.expanduser("~/Data/Helen/")
        all_annotations = os.listdir(os.path.join(self.helen_dir, "annotation"))
        random.seed(1337)  # Seed random number so shuffle is deterministic
        random.shuffle(all_annotations)
        self.annotations = all_annotations[:300] if test else all_annotations[300:]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        meta = []
        with open(os.path.join(self.helen_dir, "annotation", self.annotations[item])) as f:
            reader = csv.reader(f)
            meta = [i for i in reader]
        pic = _load_image(os.path.join(self.helen_dir, "images", "{}.jpg".format(meta[0][0])))
        w, h = pic.shape[1], pic.shape[2]
        size_multiplier = 1
        #if w > 1024 and h > 1024:
        #    size_multiplier = 2

        # Generate heatmap
        img = np.zeros([1, pic.shape[1],pic.shape[2]]).astype("float32")
        meta[1:] = [[int(round(float(i))) for i in point] for point in meta[1:]]  # Convert strings to integers for face points
        points = np.array(meta[1:])

        #size_multiplier = min(points[:1].max() - points[:1].min())
        constraint1 = int((points[:,1].max() - points[:,1].min()) // 256)
        constraint2 = int((points[:, 0].max() - points[:, 0].min()) // 256)
        size_multiplier = min(constraint1, constraint2)

        for point in meta[1:]:
            x, y = point
            s = 4 * size_multiplier
            cv2.ellipse(img[0], (x, y), (s, s), 0, 0, 360, 255, -1)

        # Generate torch image
        tot = torch.cat([torch.from_numpy(pic), torch.from_numpy(img)])

        # Crop and downsample image
        random.seed()

        w, h = tot.shape[1], tot.shape[2]
        y, x = random.choice(meta[1:])

        start = (x // size_multiplier - 128, y // size_multiplier - 128)
        if start[0] < 0 or start[0] + 256 > w or start[1] < 0 or start[1] + 256 > w:
            start = (128, 128)
        #start = (random.randint(0, w-256), random.randint(0, h-256))

        tot = F.avg_pool2d(Variable(tot, volatile=True), 1, stride=size_multiplier).data  # Downsample image
        return tot[:, start[0]:start[0] + 256, start[1]:start[1] + 256]  # Make this return statement


if __name__ == "__main__":
    data = HelenData()
    import code
    code.interact(local=locals())
