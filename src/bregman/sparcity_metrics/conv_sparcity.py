import torch


def conv_sparsity(model):
    nnz = 0
    total = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            s = m.weight.shape
            w = m.weight.view(s[0] * s[1], s[2] * s[3])
            nnz += torch.count_nonzero(torch.norm(w, p=1, dim=1) > 0).item()
            total += s[0] * s[1]
    if total == 0:
        return 0
    else:
        return nnz / total
