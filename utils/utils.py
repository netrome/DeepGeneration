import torch.nn.functional as F


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


