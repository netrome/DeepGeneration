import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.datasets as datasets
import utils.visualizer as vis
import settings
import utils.utils as u

from utils.utils import cyclic_data_iterator
from torch.autograd import Variable

dataset = u.get_data_set()
if settings.GENERATED_PATH is not None:
    dataset = datasets.GeneratedWithMaps(settings.GENERATED_PATH)
    print("Using generated data set at: {}".format(settings.GENERATED_PATH))
    if settings.CONCAT_DATA:
        dataset = torch.utils.data.ConcatDataset([dataset, u.get_data_set()])
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=settings.BATCH_SIZE,
                                          shuffle=True,
                                          pin_memory=True,
                                          drop_last=True)

R = u.create_regressor()
criterion = nn.BCEWithLogitsLoss()

if settings.CUDA:
    R.cuda()

if settings.WORKING_MODEL:
    R.load_state_dict(torch.load("working_model/R.params"))
    print("Loaded regressor model")

optimizer = torch.optim.Adam(R.parameters(), lr=settings.LEARNING_RATE)
visualizer = vis.Visualizer()
state = json.load(open("working_model/state.json", "r"))
visualizer.point = state["point"]


def update_visualization(visualizer, loss, patterns, prediction):
    prediction.data.clamp_(0, 1)
    avg = (prediction + patterns) / 2
    visualizer.update_batch(avg.data.cpu(), "real_batch")
    visualizer.update_batch(prediction.data.cpu(), "fake_batch")
    visualizer.update_loss(loss.data.cpu(), torch.ones(1))


# Main training loop

for chunk in range(settings.CHUNKS):
    print("Chunk {}/{}    ".format(chunk, settings.CHUNKS))
    for i, batch in enumerate(cyclic_data_iterator(data_loader, settings.STEPS)):
        batch = Variable(batch)
        if settings.CUDA:
            batch = batch.cuda()

        patterns = batch[:, 0, :, :].unsqueeze(1)
        targets = batch[:, 1, :, :].unsqueeze(1)

        prediction = R(patterns)
        loss = criterion(prediction, targets)

        # Perform an optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 9:
            print("Step {}/{}   ".format(i + 1, settings.STEPS), end="\r")
            update_visualization(visualizer, loss, patterns, prediction)


# Save models
print("Saving rgb layers, {}".format(time.ctime()))

torch.save(R.state_dict(), "working_model/R.params")

# Save state
state["point"] = visualizer.point
print("Saving state, {}".format(time.ctime()))
json.dump(state, open("working_model/state.json", "w"))

print("Finished with main")
