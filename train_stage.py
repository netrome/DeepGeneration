import visualizer
import settings

from utils import datasets
from utils import progressive_networks
from utils import trainer
import utils.utils as utils

from torch.autograd import Variable

import torch.utils.data
import torch

# Get utilities ---------------------------------------------------
dataset = datasets.SyntheticFullyAnnotated(settings.DATA_PATH)
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=settings.BATCH_SIZE,
                                          shuffle=True,
                                          pin_memory=True,
                                          drop_last=True)
visualizer = visualizer.Visualizer()

# Define networks -------------------------------------------------
G = progressive_networks.TrivialGenerator()
D = progressive_networks.TrivialDiscriminator()

if settings.D_PATH is not None:
    D.load_state_dict(torch.load(settings.D_PATH))
    print("Using discriminator at {}".format(settings.D_PATH))

if settings.G_PATH is not None:
    G.load_state_dict(torch.load(settings.G_PATH))
    print("Using generator at {}".format(settings.G_PATH))

# Export to cuda
if settings.CUDA:
    G.cuda()
    D.cuda()

# Add optimizer
opt_G = torch.optim.Adamax(G.parameters(), lr=settings.LEARNING_RATE)
opt_D = torch.optim.Adamax(D.parameters(), lr=settings.LEARNING_RATE)

# Train with StageTrainer
s, (c, d) = [settings.STAGE, settings.PROGRESSION[settings.STAGE]]
stage = trainer.StageTrainer(G, D, opt_G, opt_D, data_loader,
                             stage=s, conversion_depth=c, downscale_factor=d)
stage.visualize(visualizer)
for i in range(100):
    print("--- Chunk {} ---             ".format(i))
    stage.steps(100)
    stage.visualize(visualizer)

# Save networks
torch.save(G.state_dict(), "checkpoints/lastG.params")
torch.save(D.state_dict(), "checkpoints/lastD.params")

