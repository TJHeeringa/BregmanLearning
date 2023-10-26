import torch


class Null:
    r"""This is a regularizer that does nothing.

    The optimizers in this package are written such that they require a regularizer. This class is made to effectively
    have no regularization in the optimization procedure, when used.
    """
    def __call__(self, x):
        return 0

    def prox(self, x, delta=1.0):
        return x

    def sub_grad(self, v):
        return torch.zeros_like(v)
