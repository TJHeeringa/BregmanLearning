import torch


def column_sparsify(model, sparsity_level):
    """Takes all the torch.nn.Linear Modules and sets columns of the weight matrices to zero based on a Bernoulli
    random variable.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            w = m.weight.data
            mask = torch.bernoulli(sparsity_level * torch.ones(size=(w.shape[1], 1), device=w.device))
            m.weight.data.mul_(mask)
