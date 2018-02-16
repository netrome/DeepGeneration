"""
Spectral Normalization using power iteration, from
https://www.researchgate.net/publication/318572189_Spectral_Normalization_for_Generative_Adversarial_Networks
"""

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class SpectralNorm:
    def __init__(self, module, weight_name, a):
        self.weight_name = weight_name

        # Get weights and process them
        W = getattr(module, weight_name).data

        # Weight scaling
        torch.nn.init.kaiming_normal(W, a=a, mode="fan_out")  # Apply He initializer
        scale = torch.Tensor([torch.std(W)])  # Obtain scale factor
        W /= scale  # Undo initialization to obtain N(0,1) in weight tensor

        self.shape = W.shape
        assert len(self.shape) == 2 or len(self.shape) == 4  # Only supporting 2D or 4D tensors
        W = W.view(self.shape[0], -1)  # Reshape in case of 4D tensors

        # Initialize u and v with real SVD
        U, S, V = torch.svd(W)
        u, v = U[:, 0].contiguous(), V[0].contiguous()

        # Modify module to hold the correct parameters
        del module._parameters[weight_name]

        module.register_parameter("unnormalized_weight", Parameter(W))
        module.register_buffer("u", u)
        module.register_buffer("v", v)
        module.register_buffer("scale", scale)  # Registered buffers should be tensors
        module.register_forward_pre_hook(self)  # Register this as a forward hook

    def normalized_weight(self, module):
        W = getattr(module, "unnormalized_weight")
        scale = getattr(module, "scale")
        sigma = module.u.view(1, -1).mm(W.data).mm(module.v.view(-1, 1))  # Largest singular value (approx)
        # print("True: {}, estimate: {}".format(float(torch.svd(W)[1][0]), float(sigma))) - check surprisingly good
        return (W / Variable(sigma)).view(self.shape) * Variable(scale)

    def power_iteration(self, module):
        W = getattr(module, "unnormalized_weight").data
        module.v = W.t().mm(module.u.view(-1, 1))
        module.u = W.mm(module.v.view(-1, 1))
        module.v /= torch.norm(module.v)
        module.u /= torch.norm(module.u)

    def __call__(self, module, _):
        self.power_iteration(module)  # Update estimates
        setattr(module, self.weight_name, self.normalized_weight(module))  # Insert new weight


def spectral_norm(module, name='weight', a=0):
    r"""Applies spectral normalization to a weight tensor in the given module.
    .. math::
         \mathbf{w} =  \dfrac{\mathbf{W}}{\|\sigma(\mathbf{v})\|}
    Spectral normalization is a reparameterization that normalizes the magnitude
    of a weight tensor with its spectral norm, defined as the largest singular value.
    This replaces the parameter specified by `name` (e.g. "weight") with a normalized
    version of it, that is recomputed at each call.
    Spectral normalization is implemented in a similar manner as weight normalization
    via a hook that recomputes the normalized weight tensor before each :meth:`~Module.forward`
    call. It uses power iteration to approximate the spectral norm each iteration,
    as in the original paper.
    See https://www.researchgate.net/publication/318572189_Spectral_Normalization_for_Generative_Adversarial_Networks
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        reinitialize_weights (bool): if set this module will reinitialize the weights with a N(0,1) distribution to ensure a consistent dynamic range.
    Returns:
        The original module with the spectral norm hook
    Example::
        >>> m = spectral_norm(nn.Linear(20, 40), name='weight')
        Linear (20 -> 40)
    """
    SpectralNorm(module, name, a)
    return module


def supported_module(module):
    mods = [torch.nn.ConvTranspose2d, torch.nn.Conv2d, torch.nn.Linear]

    for mod in mods:
        if isinstance(module, mod):
            return True
    return False


def normalize_network(network, a=0):
    for module in network.modules():
        if supported_module(module):
            spectral_norm(module, a=a)


