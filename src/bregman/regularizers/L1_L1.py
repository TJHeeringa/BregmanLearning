import torch


class L1_L1:
    r"""This regularizer computes

    .. math::
        ||\theta||_{\ell^{1,1}} = \sum_{i}\sum_{j}|\theta_{ij}|

    where $i$ sums over the columns and $j$ over the rows.

    The associated proximal map is

    .. math::

    It is used to produce ...
    """

    def __init__(self, rc=1.0):
        self.rc = rc

    def __call__(self, x):
        return self.rc * torch.norm(torch.norm(x, p=1, dim=1), p=1).item()

    def prox(self, x, delta=1.0):
        pass

    def sub_grad(self, x):
        pass
