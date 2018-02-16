import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", help="enable cuda", action="store_true")
parser.add_argument("--gp", help="gradient penalty", action="store_true")
parser.add_argument("--lr", help="spectral normalization", action="store", default="0.0001")
parser.add_argument("--bs", help="batch size", action="store", default="8")

parser.add_argument("--stage", help="training stage", action="store", default="6")
parser.add_argument("--steps", help="training steps", action="store", default="100")
parser.add_argument("--chunks", help="training chunks", action="store", default="50")
parser.add_argument("--nd", help="discriminator iterations", action="store", default="1")
parser.add_argument("--b1", help="beta1", action="store", default="0.5")
parser.add_argument("--b2", help="beta2", action="store", default="0.99")
parser.add_argument("--load-D", help="load discriminator", action="store")
parser.add_argument("--load-G", help="load discriminator", action="store")
parser.add_argument("--wip", help="use working model", action="store_true")
parser.add_argument("--ws", help="use weight equalization", action="store_true")
parser.add_argument("--sn", help="spectral normalization in D", action="store_true")
parser.add_argument("--fade-in", help="fade in next layers", action="store_true")
parser.add_argument("--config", help="external configuration", action="store")
args = parser.parse_args()


DATA_PATH = "~/Data/DeepGeneration1"

CONFIG_PATH = args.config

WORKING_MODEL = args.wip

EQUALIZE_WEIGHTS = args.ws

SPECTRAL_NORM = args.sn

BATCH_SIZE = int(args.bs)

DISCRIMINATOR_ITERATIONS = int(args.nd)

LEARNING_RATE = float(args.lr)
if EQUALIZE_WEIGHTS:
    LEARNING_RATE *= 20

BETAS = (float(args.b1), float(args.b2))

CUDA = args.cuda

GRADIENT_PENALTY = args.gp

FADE_IN = args.fade_in

STAGE = int(args.stage)

CHUNKS = int(args.chunks)
STEPS = int(args.steps)

# Stage -> depth + downscale factor
PROGRESSION = {
    1: (112, 32),
    2: (96, 16),
    3: (80, 8),
    4: (64, 4),
    5: (32, 2),
    6: (16, 1),
}


def sync_settings():
    args.bs = BATCH_SIZE
    args.nd = DISCRIMINATOR_ITERATIONS
    args.lr = LEARNING_RATE
    args.b1, args.b2 = BETAS[0], BETAS[1]
    args.cuda = CUDA
    args.gp = GRADIENT_PENALTY
    args.fade_in = FADE_IN
    args.stage = STAGE
    args.chunks = CHUNKS
    args.steps = STEPS
    args.wip = WORKING_MODEL
