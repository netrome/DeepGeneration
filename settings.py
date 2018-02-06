import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", help="enable cuda", action="store_true")
parser.add_argument("--gp", help="gradient penalty", action="store_true")
parser.add_argument("--sn", help="spectral normalization", action="store_true")
parser.add_argument("--pxnorm", help="normalize feature maps", action="store_true")
parser.add_argument("--lr", help="spectral normalization", action="store", default=0.0001)
parser.add_argument("--rsn", help="reinitialize spectral normalization", action="store_true")
parser.add_argument("--ws", help="weight scaling", action="store_true")

parser.add_argument("--stage", help="training stage", action="store", default=6)
parser.add_argument("--steps", help="training steps", action="store", default=100)
parser.add_argument("--iterations", help="training iterations", action="store", default=50)
parser.add_argument("--load-D", help="load discriminator", action="store")
parser.add_argument("--load-G", help="load discriminator", action="store")
args = parser.parse_args()


DATA_PATH = "~/Data/DeepGeneration1"

D_PATH = args.load_D
G_PATH = args.load_G

BATCH_SIZE = 7

DISCRIMINATOR_ITERATIONS = 1

EPOCHS = 1000

NORMALIZE = args.pxnorm

LEARNING_RATE = float(args.lr)

CUDA = args.cuda

GRADIENT_PENALTY = args.gp

SPECTRAL_NORM = args.sn
REINITIALIZE_SPECTRAL_NORM = args.rsn

WEIGHT_SCALING = args.ws

STAGE = int(args.stage)

ITERATIONS = int(args.iterations)
STEPS = int(args.steps)

# Stage -> depth + downscale factor
PROGRESSION = {
    1: (256, 32),
    2: (256, 16),
    3: (128, 8),
    4: (128, 4),
    5: (128, 2),
    6: (128, 1),
}
