import torch
import math


def uniform(model, r0, r1, ltype=torch.nn.Linear):
    for m in model.modules():
        if isinstance(m, ltype):
            if hasattr(m, "bias") and not (m.bias is None):
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound0 = r0 / math.sqrt(fan_in)
                bound1 = r1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound0, bound1)


def constant(model, r):
    for m in model.modules():
        if isinstance(m, torch.torch.nn.Linear):
            if type(m) == torch.nn.Linear:
                torch.nn.init.constant_(m.bias, r)
