import torch
import math


def sparsify_(model, sparsity, ltype=torch.nn.Linear, conv_group=True, row_group=False):
    for m in model.modules():
        if not isinstance(m, ltype):
            continue

        elif (isinstance(m, torch.nn.Linear) and not row_group) or (isinstance(m, torch.nn.Conv2d) and not conv_group):
            s_loc = sparsity
            mask = torch.bernoulli(s_loc * torch.ones_like(m.weight))
            m.weight.data.mul_(mask)

        elif isinstance(m, torch.nn.Linear):  # row sparsity
            s_loc = sparsity
            w = m.weight.data
            mask = torch.bernoulli(s_loc * torch.ones(size=(w.shape[0], 1), device=w.device))
            m.weight.data.mul_(mask)

        elif isinstance(m, torch.nn.Conv2d):  # kernel sparsity
            s_loc = sparsity
            w = m.weight.data
            n = w.shape[0] * w.shape[1]

            # assign mask
            mask = torch.zeros(n, 1, device=w.device)
            idx = torch.randint(low=0, high=n, size=(math.ceil(n * s_loc),))
            mask[idx] = 1

            # multiply with mask
            c = w.view(w.shape[0] * w.shape[1], -1)
            m.weight.data = mask.mul(c).view(w.shape)
