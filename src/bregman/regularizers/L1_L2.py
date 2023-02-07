import torch
import math


class L1_L2:
    r"""This regularizer computes

    .. math::
        ||\theta||_{\ell^{1,2}} = \sum_{i}\sqrt{\sum_{j}|\theta_{ij}|^2}

    or

    .. math::
        ||\theta||_{\ell^{1,2}} = \sum_{j}\sqrt{\sum_{i}|\theta_{ij}|^2}

    where $i$ sums over the columns and $j$ over the rows, when the given direction parameter is row or column respectively.

    The associated proximal map is

    .. math::
        prox_{\delta J}(v) = (prox_{\delta ||\theta_{1:}||_2}(v_1),...,prox_{\delta ||\theta_{n:}||_2}(v_n))

    with

    .. math::
        prox_{\delta ||\theta_{i:}||_2}(v_i) = \begin{cases}
            1-\frac{\delta}{||v_i||_2} & ||v_i||_2\geq \delta \\
            0 & ||v_i||_2< \delta
        \end{cases}

    It is used to produce row-sparse or column-sparse matrices.
    """

    def __init__(self, rc=1.0, direction="row"):
        assert direction in ["row", "column"]
        if direction == "row":
            self.dim = 1
        else:
            self.dim = 0
        self.rc = rc

    def __call__(self, x):
        return self.rc * math.sqrt(x.shape[-1]) * torch.norm(torch.norm(x, p=2, dim=self.dim), p=1).item()

    def prox(self, x, delta=1.0):
        thresh = delta * self.rc * math.sqrt(x.shape[-1])

        ret = torch.clone(x)
        nx = torch.norm(x, p=2, dim=self.dim).view(x.shape[0], 1)

        ind = torch.where((nx != 0))[0]

        ret[ind] = x[ind] * torch.clamp(1 - torch.clamp(thresh / nx[ind], max=1), min=0)
        return ret

    def sub_grad(self, x):
        thresh = self.rc * math.sqrt(x.shape[-1])

        nx = torch.norm(x, p=2, dim=self.dim).view(x.shape[0], 1)
        ind = torch.where((nx != 0))[0]
        ret = torch.clone(x)
        ret[ind] = x[ind] / nx[ind]
        return thresh * ret
