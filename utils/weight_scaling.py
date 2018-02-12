"""
Live weight scaling, as in PGAN
"""

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class RuntimeWieghtScale:
    def __init__(self, module, weight_name, a):
        # Initialize weights with N(0,1)
        self.weight_name = weight_name
        W = getattr(module, self.weight_name)
        del module._parameters[self.weight_name]

        torch.nn.init.kaiming_normal(W, a=a, mode="fan_out")  # Apply He initializer
        scale = torch.std(W)  # Obtain scale factor
        W /= scale  # Undo initialization to obtain N(0,1) in weight tensor

        # Detach to avoid unwanted dependencies in the graph
        W = W.detach()
        scale = scale.detach()

        module.register_parameter("equalized_weight", Parameter(W.data))
        module.register_buffer("scale", scale.data)  # Registered buffers should be tensors
        module.register_forward_pre_hook(self)

    def __call__(self, module, _):
        # Scale weights dynamically
        setattr(module, self.weight_name, module.equalized_weight * Variable(module.scale))


def equalized_weights(module, name="weight", a=0):
    RuntimeWieghtScale(module, weight_name=name, a=a)
    return module


def supported_module(module):
    mods = [torch.nn.ConvTranspose2d, torch.nn.Conv2d, torch.nn.Linear]

    for mod in mods:
        if isinstance(module, mod):
            return True
    return False


def scale_network(network, a=0):
    for module in network.modules():
        if supported_module(module):
            equalized_weights(module, a=a)



