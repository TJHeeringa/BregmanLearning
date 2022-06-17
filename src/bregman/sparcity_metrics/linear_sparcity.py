import torch


def linear_sparsity(model):
    numel = 0
    nnz = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            a = m.weight
            numel_loc = a.data.numel()
            numel += numel_loc
            nnz += torch.count_nonzero(a.data).item()
    return nnz / numel
