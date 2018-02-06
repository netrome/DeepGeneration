""" Initializes new networks and RGB layers, and stores them in working_model """
import torch
from utils import progressive_networks
import torch.nn as nn
import settings

import visualizer

G = progressive_networks.TrivialGenerator()
D = progressive_networks.TrivialDiscriminator()

visualizer = visualizer.Visualizer()
visualizer.initiate_windows()

torch.save(G.state_dict(), "working_model/G.params")
torch.save(D.state_dict(), "working_model/D.params")

for i in settings.PROGRESSION:
    c, d = settings.PROGRESSION[i]
    to_rgb = nn.Conv2d(c, 2, 1)
    from_rgb = nn.Conv2d(2, c, 1)

    torch.save(to_rgb.state_dict(), "working_model/toRGB{}.params".format(i))
    torch.save(from_rgb.state_dict(), "working_model/fromRGB{}.params".format(i))

print("Saved networks and RGB layers in ./working_model")
