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

# Export to cuda
if settings.CUDA:
    G.cuda()
    D.cuda()

# Add optimizer
opt_G = torch.optim.Adam(G.parameters(), lr=settings.LEARNING_RATE)
opt_D = torch.optim.Adam(D.parameters(), lr=settings.LEARNING_RATE)

# Train with StageTrainer
stage = trainer.StageTrainer(G, D, opt_G, opt_D, data_loader, stage=6, conversion_depth=16)
stage.visualize(visualizer)
for i in range(10):
    print("Chunk {}".format(i))
    stage.steps(100)
    stage.visualize(visualizer)

