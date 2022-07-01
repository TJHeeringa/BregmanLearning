import torch


def network_sparsity(model):
    """This metric goes over all the Linear Modules in the given model and computes for each of the weight matrices the
    number of zero and nonzero parameters. It returns the ratio of the nonzero entries compared with the total entries.
    """
    numel = 0
    nnz = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            a = m.weight
            numel_loc = a.data.numel()
            numel += numel_loc
            nnz += torch.count_nonzero(a.data).item()
    return nnz / numel
