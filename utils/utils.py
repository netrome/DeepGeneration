import torch.nn.functional as F


def downsample_tensor(tensor, factor):
    return F.avg_pool2d(tensor, kernel_size=factor, stride=factor)
