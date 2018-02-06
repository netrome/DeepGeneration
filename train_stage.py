import visualizer
import settings

from utils import datasets
from utils import progressive_networks
from utils import trainer
import utils.utils as utils

from torch.autograd import Variable

import torch.utils.data
import torch

import json

print("\nInitiating training with the following setting ----")
print(json.dumps(vars(settings.args), sort_keys=True, indent=4))
print("---------------------------------------------------")
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

if settings.WORKING_MODEL:
    print("Using model parameters in ./working_model")
    G.load_state_dict(torch.load("working_model/G.params"))
    D.load_state_dict(torch.load("working_model/D.params"))

#if settings.D_PATH is not None:
#    D.load_state_dict(torch.load(settings.D_PATH))
#    print("Using discriminator at {}".format(settings.D_PATH))
#
#if settings.G_PATH is not None:
#    G.load_state_dict(torch.load(settings.G_PATH))
#    print("Using generator at {}".format(settings.G_PATH))

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
if settings.WORKING_MODEL:
    stage.toRGB.load_state_dict(torch.load("working_model/toRGB{}.params".format(s)))
    stage.fromRGB.load_state_dict(torch.load("working_model/fromRGB{}.params".format(s)))
    print("Loaded RGB layers too")

stage.visualize(visualizer)
for i in range(settings.CHUNKS):
    print("Chunk {}                   ".format(i))
    stage.steps(settings.STEPS)
    stage.visualize(visualizer)

# Save networks
to_rgb, from_rgb = stage.get_rgb_layers()
torch.save(to_rgb.state_dict(), "working_model/toRGB{}.params".format(s))
torch.save(from_rgb.state_dict(), "working_model/toRGB{}.params".format(s))
torch.save(G.state_dict(), "working_model/G.params")
torch.save(D.state_dict(), "working_model/D.params")

