import torch


def sparsify(model: torch.nn.Module, density_level: float):
    r"""Takes all the torch.nn.Linear Modules and sets rows of the weight matrices to zero based on a Bernoulli
    random variable such that in the end

    .. math::
        \#(\text{nonzero rows}) / \#\text{rows} = \lceil \#\text{rows} * \text{density\_level} \rceil.

    """
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            w = m.weight.data

            # get number of rows that need to be zeroed
            num_rows, num_cols = w.shape
            zero_row_count = torch.ceil(num_rows * (1-density_level))

            # create indices for the rows that will be zeroed
            indices = []
            while len(indices) < zero_row_count:
                random_index = torch.randint(0, num_rows)
                if random_index not in indices:
                    indices.append(random_index)

            # create mask that does the zeroing based on the indices
            mask = torch.ones(w.shape)
            for index in indices:
                mask[index] = torch.zeros((1, num_cols))

            # apply the mask
            m.weight.data.mul_(mask)
