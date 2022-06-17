import torch


def net_sparsity(model):
    numel = 0
    nnz = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            a = m.weight
            numel_loc = a.data.numel()
            numel += numel_loc
            nnz += torch.count_nonzero(a.data).item()
    return nnz / numel
