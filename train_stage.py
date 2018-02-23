import json

import torch
import torch.cuda
import torch.utils.data

import settings
import time
from utils.visualizer import Visualizer
import utils.weight_scaling as ws
import utils.spectral_norm as sn
import utils.trainer as trainer

import utils.utils as u

import gc


def main():
    print("\nInitiating training with the following setting ----")
    print(json.dumps(vars(settings.args), sort_keys=True, indent=4))
    print("---------------------------------------------------")
    # Get utilities ---------------------------------------------------
    dataset = u.get_data_set()
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=settings.BATCH_SIZE,
                                              shuffle=True,
                                              pin_memory=True,
                                              drop_last=True)
    visualizer = Visualizer()
    state = json.load(open("working_model/state.json", "r"))
    visualizer.point = state["point"]

    # Define networks -------------------------------------------------
    G = u.create_generator()
    D = u.create_discriminator()

    if settings.EQUALIZE_WEIGHTS:
        ws.scale_network(D, 0.2)
        ws.scale_network(G, 0.2)

    if settings.SPECTRAL_NORM:
        sn.normalize_network(D, 0.2)

    if settings.WORKING_MODEL:
        print("Using model parameters in ./working_model")
        G.load_state_dict(torch.load("working_model/G.params"))
        D.load_state_dict(torch.load("working_model/D.params"))

    # Train with StageTrainer or FadeInTrainer
    s, (c, d) = [settings.STAGE, settings.PROGRESSION[settings.STAGE]]
    if settings.FADE_IN:
        print("Freezing in next layer")
        c = settings.PROGRESSION[settings.STAGE + 1][0]
        d = int(d/2)
        G.freeze_until(s)
        #D.freeze_until(s)
        s += 1

    # Freeze idle layers - did not stop vlad
    #G.freeze_idle(s)
    #D.freeze_idle(s)

    stage = trainer.StageTrainer(G, D, data_loader,
                                 stage=s, conversion_depth=c, downscale_factor=d)
    stage.pred_real += state["pred_real"]
    stage.pred_fake += state["pred_fake"]

    if settings.WORKING_MODEL:
        stage.toRGB.load_state_dict(torch.load("working_model/toRGB{}.params".format(s)))
        stage.fromRGB.load_state_dict(torch.load("working_model/fromRGB{}.params".format(s)))
        print("Loaded RGB layers too")

    stage.visualize(visualizer)
    for i in range(settings.CHUNKS):
        print("Chunk {}, stage {}, fade in: {}, GPU memory {}               "
              .format(i, settings.STAGE, settings.FADE_IN, 1337))
        stage.steps(settings.STEPS)
        gc.collect()  # Prevent memory leaks (?)
        #torch.cuda.empty_cache()  - Made no difference
        state["history_real"].append(float(stage.pred_real))
        state["history_fake"].append(float(stage.pred_fake))
        if settings.WORKING_MODEL:
            print("Saved timelapse visualization")
            stage.save_fake_reference_batch(visualizer.point)
        stage.visualize(visualizer)

    # Save networks
    """
    if settings.FADE_IN:
        to_rgb, from_rgb, next_to_rgb, next_from_rgb = stage.get_rgb_layers()
        print("Saving extra rgb layers, {}".format(time.ctime()))
        torch.save(next_to_rgb.state_dict(), "working_model/toRGB{}.params".format(s + 1))
        torch.save(next_from_rgb.state_dict(), "working_model/fromRGB{}.params".format(s + 1))
    else:
        to_rgb, from_rgb = stage.get_rgb_layers()
    """
    to_rgb, from_rgb = stage.get_rgb_layers()
    print("Saving rgb layers, {}".format(time.ctime()))

    torch.save(to_rgb.state_dict(), "working_model/toRGB{}.params".format(s))
    torch.save(from_rgb.state_dict(), "working_model/fromRGB{}.params".format(s))
    print("Saving networks, {}".format(time.ctime()))
    G.unfreeze_all()
    D.unfreeze_all()
    torch.save(G.state_dict(), "working_model/G.params")
    torch.save(D.state_dict(), "working_model/D.params")

    # Save state
    state["point"] = visualizer.point
    state["pred_real"] = float(stage.pred_real)
    state["pred_fake"] = float(stage.pred_fake)
    print("Saving state, {}".format(time.ctime()))
    json.dump(state, open("working_model/state.json", "w"))

    # Save optimizer state
    #opt_G = stage.opt_G
    #opt_D = stage.opt_D

    #print("Saving optimizer state, {}".format(time.ctime()))
    #torch.save(opt_G.state_dict(), "working_model/optG.state")
    #torch.save(opt_D.state_dict(), "working_model/optD.state")
    print("Finished with main")

if __name__ == "__main__":
    main()
