""" Take a look at the state of a saved optimizer """

import torch
import sys

state_dict = torch.load(sys.argv[1])

for key in state_dict:
    print(key)

print("Parameter groups: -------------")
print(state_dict["param_groups"])

print("State norms")
for key in state_dict["state"]:
    param_dict = state_dict["state"][key]
    print(torch.norm(param_dict["exp_avg"]))
    print(torch.norm(param_dict["exp_inf"]))

