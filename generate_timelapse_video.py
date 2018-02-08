import cv2
from skvideo.io import FFmpegWriter
import numpy as np
import re

import glob

img = []

files = [i for i in glob.glob("working_model/timelapse/*.png")]
files.sort(key=lambda x: int(re.findall("\d+", x)[0]))

out = FFmpegWriter("working_model/timelapse/timelapse.mp4")

for f in files:
    img = cv2.resize(cv2.imread(f), (1042, 262))
    out.writeFrame(img)

out.close()

