import torch


def node_sparsity(model):
    ret = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            a = m.weight

            nnz = torch.count_nonzero(torch.norm(a.data, p=2, dim=1)).item()
            numel_loc = a.shape[0]
            ret.append(nnz / numel_loc)
    return ret
