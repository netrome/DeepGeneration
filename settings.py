import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", help="enable cuda", action="store_true")
parser.add_argument("--gp", help="gradient penalty", action="store_true")
parser.add_argument("--sn", help="spectral normalization", action="store_true")
parser.add_argument("--pxnorm", help="normalize feature maps", action="store_true")
parser.add_argument("--lr", help="spectral normalization", action="store")
parser.add_argument("--rsn", help="reinitialize spectral normalization", action="store_true")
parser.add_argument("--ws", help="weight scaling", action="store_true")
args = parser.parse_args()


DATA_PATH = "~/Data/DeepGeneration1"

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