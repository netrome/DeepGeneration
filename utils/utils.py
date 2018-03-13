import settings

import torch.nn.functional as F
import torch.nn as nn
import torch

from torch.nn import Parameter

from utils.networks import TrivialEncoderLight, ImageToImage
from utils.progressive_networks import TrivialGeneratorLight, TrivialDiscriminatorLight

import utils.datasets as data


def downsample_tensor(tensor, factor):
    return F.avg_pool2d(tensor, kernel_size=factor, stride=factor)


def cyclic_data_iterator(data_loader, n=100):
    i = 0
    while i < n:
        for batch in data_loader:
            i += 1
            yield batch
            if i >= n:
                break


def near_identity_weight_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        if m.weight.shape[2] == 3:
            identity = torch.eye(m.weight.shape[0], m.weight.shape[1])
            m.weight.data = torch.randn(m.weight.shape) * 1e-6
            m.weight.data[:, :, 1, 1] = identity

        if m.weight.shape[2] == 2:
            avg = torch.ones(m.weight.shape) / \
                  (m.weight.shape[1] + m.weight.shape[2] + m.weight.shape[3])
            m.weight.data = torch.randn(m.weight.shape) * 1e-6 + avg


def _create_network(network_class):
    net = network_class()
    if settings.CUDA:
        net.cuda()
    return net


def create_generator():
    return _create_network(settings.GENERATOR)


def create_discriminator():
    return _create_network(settings.DISCRIMINATOR)


def create_encoder():
    return _create_network(settings.ENCODER)


def create_regressor():
    return _create_network(settings.REGRESSOR)


def get_data_set():
    if settings.REAL_DATA:
        return data.DeepGazeData(settings.TEST_DATA)
    elif settings.HELEN_DATA:
        return data.HelenData(settings.TEST_DATA)
    else:
        return data.SyntheticFullyAnnotated(settings.DATA_PATH)

