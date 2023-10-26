from .sparsify_column import sparsify_column
from .sparsify_full import sparsify_full
from .sparsify_row import sparsify_row


def sparsify(model, sparsity_level, direction="", allow_zero_matrices=False):
    if direction == "row":
        sparsify_row(model, sparsity_level, allow_zero_matrices=allow_zero_matrices)
    elif direction == "column":
        sparsify_column(model, sparsity_level, allow_zero_matrices=allow_zero_matrices)
    else:
        sparsify_full(model, sparsity_level, allow_zero_matrices=allow_zero_matrices)
