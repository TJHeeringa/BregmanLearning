import torch


def node_sparsity(model, direction="row"):
    """This metric goes over all the Linear Modules in the given model and computes for each of the weight matrices the
    row or column sparsity depending on the direction parameter.
    """
    assert direction in ["row", "column"]
    if direction == "row":
        norm_dim = 0
        size_dim = 1
    else:
        norm_dim = 1
        size_dim = 0
    ret = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            a = m.weight

            nnz = torch.count_nonzero(torch.norm(a.data, p=2, dim=norm_dim)).item()
            numel_loc = a.shape[size_dim]
            ret.append(nnz / numel_loc)
    return ret
