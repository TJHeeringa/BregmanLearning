from abc import ABC, abstractmethod
import math

import torch


class BaseRegularizer(ABC):
    r"""Bregman requires regularizers of a specific form. This specifies which methods need to be implemented."""

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def prox(self, x, delta=1.0):
        pass

    @abstractmethod
    def sub_grad(self, v):
        pass


class Null(BaseRegularizer):
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


class L1(BaseRegularizer):
    r"""This regularizer computes

    .. math::
        ||\theta||_{\ell^2} = \sum_{i}|\theta_i|

    The associated proximal map is

    .. math::
        prox(\theta)_i = \sgn(\theta_i) min(0, |\theta_i|-\delta\lambda)

    It is used to produce sparse vectors for e.g. biases or skip layers.
    """

    def __init__(self, rc=1.0):
        self.rc = rc

    def __call__(self, x):
        return self.rc * torch.norm(x, p=1).item()

    def prox(self, x, delta=1.0):
        return torch.sign(x) * torch.clamp(torch.abs(x) - (delta * self.rc), min=0)

    def sub_grad(self, v):
        return self.rc * torch.sign(v)


class L11(BaseRegularizer):
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


class L12(BaseRegularizer):
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

    @torch.no_grad()
    def prox(self, x, delta=1.0):
        thresh = delta * self.rc * math.sqrt(x.shape[-1])

        ret = torch.clone(x).detach()
        nx = torch.norm(x, p=2, dim=self.dim).view(x.shape[0], 1)

        ind = torch.where((nx != 0))[0]

        # ret[ind] = x[ind] * torch.clamp(1 - torch.clamp(thresh / nx[ind], max=1), min=0)
        ret[ind] *= torch.clamp(1 - thresh / nx[ind], min=0)
        return ret

    @torch.no_grad()
    def sub_grad(self, x):
        thresh = self.rc * math.sqrt(x.shape[-1])

        nx = torch.norm(x, p=2, dim=self.dim).view(x.shape[0], 1)
        ind = torch.where((nx != 0))[0]
        ret = torch.clone(x)
        ret[ind] = x[ind] / nx[ind]
        return thresh * ret


class Nuclear(BaseRegularizer):
    r"""This regularizer computes

    .. math::
        ||\theta||_{1} = \sum_{i}|\sigma_i(\theta)|

    where $\sigma_i(\theta)$ is the $i$-th singular value of $\theta$. The
    associated proximal map is

    .. math::
        prox(\theta) = U diag(\prox_{||\cdot||_1}(\sigma)) V^T

    where $\theta= U \Sigma V^T$ is the SVD of $\theta$ and $\sigma_i=\Sigma_ii$.
    """

    def __init__(self, rc=1.0):
        self.rc = rc

    def __call__(self, x):
        return self.rc * torch.norm(x, "nuc")

    @torch.no_grad()
    def prox(self, x, delta=1.0):
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        return U @ torch.diag_embed(torch.clamp(S - (delta * self.rc), min=0)) @ Vh

    @torch.no_grad()
    def sub_grad(self, v):
        U, S, Vh = torch.linalg.svd(v, full_matrices=False)
        return self.rc * U @ Vh


class SoftBernoulli(BaseRegularizer):
    def __init__(self, rc=1.0):
        self.rc = rc

    def prox(self, x, delta=1.0):
        return torch.sign(x) * torch.max(
            torch.clamp(torch.abs(x) - (delta * self.rc), min=0),
            torch.bernoulli(0.01 * torch.ones_like(x)),
        )

    def sub_grad(self, v):
        return self.rc * torch.sign(v)
