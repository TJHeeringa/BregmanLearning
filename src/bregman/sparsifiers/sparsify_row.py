import torch


def sparsify_row(model, sparsity_level, allow_zero_matrices=False):
    """Takes all the torch.nn.Linear Modules and sets rows of the weight matrices to zero based on a Bernoulli
    random variable.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            w = m.weight.data
            mask = torch.bernoulli(sparsity_level * torch.ones(size=(w.shape[0], 1), device=w.device))
            if not allow_zero_matrices and mask.sum() == 0:
                random_index = torch.randint(0, w.shape[0], (1,))
                mask[random_index] = 1
            m.weight.data.mul_(mask)
