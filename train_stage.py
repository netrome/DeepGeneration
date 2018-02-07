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
state = json.load(open("working_model/state.json", "r"))
visualizer.point = state["point"]


# Define networks -------------------------------------------------
G = progressive_networks.TrivialGenerator()
D = progressive_networks.TrivialDiscriminator()

if settings.WORKING_MODEL:
    print("Using model parameters in ./working_model")
    G.load_state_dict(torch.load("working_model/G.params"))
    D.load_state_dict(torch.load("working_model/D.params"))

# Export to cuda
if settings.CUDA:
    G.cuda()
    D.cuda()

# Train with StageTrainer or FadeInTrainer
s, (c, d) = [settings.STAGE, settings.PROGRESSION[settings.STAGE]]
if settings.FADE_IN:
    print("Fading in next layer")
    next_cd = settings.PROGRESSION[settings.STAGE + 1][0]
    increment = 1/(settings.CHUNKS * settings.STEPS)
    stage = trainer.FadeInTrainer(G, D, data_loader, stage=s, conversion_depth=c,
                                  downscale_factor=int(d/2), next_cd=next_cd, increment=increment)
else:
    stage = trainer.StageTrainer(G, D, data_loader,
                                 stage=s, conversion_depth=c, downscale_factor=d)
stage.pred_real += state["pred_real"]
stage.pred_fake += state["pred_fake"]

if settings.WORKING_MODEL:
    print(s)
    stage.toRGB.load_state_dict(torch.load("working_model/toRGB{}.params".format(s)))
    stage.fromRGB.load_state_dict(torch.load("working_model/fromRGB{}.params".format(s)))
    print("Loaded RGB layers too")

stage.visualize(visualizer)
for i in range(settings.CHUNKS):
    print("Chunk {}, stage {}, fade in: {}                   ".format(i, settings.STAGE, settings.FADE_IN))
    stage.steps(settings.STEPS)
    stage.visualize(visualizer)

# Save networks
if settings.FADE_IN:
    to_rgb, from_rgb, next_to_rgb, next_from_rgb = stage.get_rgb_layers()
    torch.save(next_to_rgb.state_dict(), "working_model/toRGB{}.params".format(s + 1))
    torch.save(next_from_rgb.state_dict(), "working_model/fromRGB{}.params".format(s + 1))
else:
    to_rgb, from_rgb = stage.get_rgb_layers()

torch.save(to_rgb.state_dict(), "working_model/toRGB{}.params".format(s))
torch.save(from_rgb.state_dict(), "working_model/fromRGB{}.params".format(s))
torch.save(G.state_dict(), "working_model/G.params")
torch.save(D.state_dict(), "working_model/D.params")

# Save state
state = {
    "point": visualizer.point,
    "pred_real": float(stage.pred_real),
    "pred_fake": float(stage.pred_fake),
}
json.dump(state, open("working_model/state.json", "w"))
