import torch


def full_sparsify(model, sparsity_level):
    """Takes all the torch.nn.Linear Modules and sets elements of the weight matrices to zero based on a Bernoulli
    random variable.
    """

    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            mask = torch.bernoulli(sparsity_level * torch.ones_like(m.weight))
            m.weight.data.mul_(mask)
