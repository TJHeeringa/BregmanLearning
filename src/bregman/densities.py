import torch


def network_density(model: torch.nn.Module, absolute: bool = False):
    """This metric goes over all the Linear Modules in the given model and computes for each of the weight matrices the
    number of zero and nonzero parameters. It returns the ratio of the nonzero entries compared with the total entries.

    Args:
        model: neural network you want the network density of
        absolute: if true, it will return the number of nonzero entries instead.
    """
    num_elem = 0
    num_nonzero = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            data = m.weight.data
            num_elem += data.numel()
            num_nonzero += torch.count_nonzero(data).item()
    if absolute:
        return num_nonzero
    return num_nonzero / num_elem


def row_density(model: torch.nn.Module, absolute: bool = False):
    """This metric goes over all the Linear Modules in the given model and computes for each of the weight matrices the
    row density.

    Args:
        model: neural network you want the row density of
        absolute: if true, it will return the number of nonzero rows instead.
    """
    metric_per_row = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            a = m.weight

            num_nonzero_rows = torch.count_nonzero(torch.norm(a.data, p=2, dim=0)).item()
            num_rows = a.shape[1]

            if absolute:
                metric_per_row.append(num_nonzero_rows)
            else:
                metric_per_row.append(num_nonzero_rows / num_rows)
    return metric_per_row


def column_density(model: torch.nn.Module, absolute: bool = False):
    """This metric goes over all the Linear Modules in the given model and computes for each of the weight matrices the
    column density.

    Args:
        model: neural network you want the row density of
        absolute: if true, it will return the number of nonzero cols instead.
    """
    metric_per_col = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            a = m.weight

            num_nonzero_cols = torch.count_nonzero(torch.norm(a.data, p=2, dim=1)).item()
            num_cols = a.shape[0]

            if absolute:
                metric_per_col.append(num_nonzero_cols)
            else:
                metric_per_col.append(num_nonzero_cols / num_cols)
    return metric_per_col

