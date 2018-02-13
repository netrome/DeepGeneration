""" Initializes new networks and RGB layers, and stores them in working_model """
import json

import torch
import torch.nn as nn

import settings
from utils import progressive_networks
from utils.visualizer import Visualizer
import utils.weight_scaling as ws


def main():
    G = progressive_networks.SamplingGeneratorLight()
    D = progressive_networks.SamplingDiscriminatorLight()

    if settings.EQUALIZE_WEIGHTS:
        ws.scale_network(D, 0.2)
        ws.scale_network(G, 0.2)

    opt_G = torch.optim.Adamax(G.parameters(), settings.LEARNING_RATE, betas=settings.BETAS)
    opt_D = torch.optim.Adamax(D.parameters(), settings.LEARNING_RATE, betas=settings.BETAS)

    visualizer = Visualizer()
    visualizer.initiate_windows()

    torch.save(G.state_dict(), "working_model/G.params")
    torch.save(D.state_dict(), "working_model/D.params")

    torch.save(opt_G.state_dict(), "working_model/optG.state")
    torch.save(opt_D.state_dict(), "working_model/optD.state")

    for i in settings.PROGRESSION:
        c, d = settings.PROGRESSION[i]
        to_rgb = nn.Conv2d(c, 2, 1)
        from_rgb = nn.Conv2d(2, c, 1)

        torch.save(to_rgb.state_dict(), "working_model/toRGB{}.params".format(i))
        torch.save(from_rgb.state_dict(), "working_model/fromRGB{}.params".format(i))

    # Initialize state
    state = {"point": 0, "pred_real": 0, "pred_fake": 0, "history_real": [], "history_fake": []}
    json.dump(state, open("working_model/state.json", "w"))
    # -----------------
    print("Saved networks and RGB layers in ./working_model")

if __name__ == "__main__":
    main()

