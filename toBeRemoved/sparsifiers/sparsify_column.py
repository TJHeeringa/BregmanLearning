import torch


def sparsify_column(model, sparsity_level, allow_zero_matrices=False):
    """Takes all the torch.nn.Linear Modules and sets columns of the weight matrices to zero based on a Bernoulli
    random variable.
    """
    # TODO: throws errors for non-square weight matrices on w.shape[1]; fix
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            w = m.weight.data
            if not allow_zero_matrices and mask.sum() == 0:
                random_index = torch.randint(0, w.shape[0], (1,))
                mask[random_index] = 1
            mask = torch.bernoulli(sparsity_level * torch.ones(size=(w.shape[1], 1), device=w.device))
            m.weight.data.mul_(mask)
