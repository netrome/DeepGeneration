import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", help="enable cuda", action="store_true")
parser.add_argument("--gp", help="gradient penalty", action="store_true")
parser.add_argument("--lr", help="spectral normalization", action="store", default="0.0001")
parser.add_argument("--bs", help="batch size", action="store", default="10")

parser.add_argument("--stage", help="training stage", action="store", default="6")
parser.add_argument("--steps", help="training steps", action="store", default="100")
parser.add_argument("--chunks", help="training chunks", action="store", default="50")
parser.add_argument("--nd", help="discriminator iterations", action="store", default="1")
parser.add_argument("--load-D", help="load discriminator", action="store")
parser.add_argument("--load-G", help="load discriminator", action="store")
parser.add_argument("--wip", help="use working model", action="store_true")
parser.add_argument("--fade-in", help="fade in next layers", action="store_true")
args = parser.parse_args()


DATA_PATH = "~/Data/DeepGeneration1"

WORKING_MODEL = args.wip

BATCH_SIZE = int(args.bs)

DISCRIMINATOR_ITERATIONS = int(args.nd)

LEARNING_RATE = float(args.lr)

CUDA = args.cuda

GRADIENT_PENALTY = args.gp

FADE_IN = args.fade_in

STAGE = int(args.stage)

CHUNKS = int(args.chunks)
STEPS = int(args.steps)

# Stage -> depth + downscale factor
PROGRESSION = {
    1: (256, 32),
    2: (256, 16),
    3: (128, 8),
    4: (64, 4),
    5: (32, 2),
    6: (16, 1),
}
