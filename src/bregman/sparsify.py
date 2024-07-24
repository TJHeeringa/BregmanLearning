import math

import torch


@torch.no_grad()
def sparsify(model: torch.nn.Module, density_level: float, spectral_sparsity: bool = True):
    r"""Takes all the torch.nn.Linear Modules and sets rows of the weight matrices to zero based on a Bernoulli
    random variable such that in the end

    .. math::
        \#(\text{nonzero rows}) / \#\text{rows} = \lceil \#\text{rows} * \text{density\_level} \rceil.

    """
    weight_matrix_counter = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            weight_matrix_counter += 1
            w = m.weight.data
            num_rows, num_cols = w.shape

            if spectral_sparsity:
                if weight_matrix_counter == len(model.encoder_layers) - 1:
                    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
                    init_weights = torch.zeros((num_rows,))
                    init_weights[0] = 1
                    m.weight.data = init_weights * U @ Vh
                    continue

            zero_row_count = math.floor(num_rows * (1 - density_level))

            if zero_row_count > 0:
                # create indices for the rows that will be zeroed
                indices = torch.multinomial(torch.ones((num_rows,)), zero_row_count, replacement=False)

                # create mask that does the zeroing based on the indices
                mask = torch.ones(w.shape)
                mask[indices] = torch.zeros((1, num_cols))

                # apply the mask
                m.weight.data.mul_(mask)
