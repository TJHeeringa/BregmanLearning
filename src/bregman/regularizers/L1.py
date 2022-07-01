import torch


class L1:
    r"""This regularizer computes

    .. math::
        ||\theta||_{\ell^2} = \sum_{i}|\theta_i|

    The associated proximal map is

    .. math::
        prox(\theta)_i = \sgn(\theta_i) min(0, |\theta_i|-\delta\lambda)

    It is used to produce sparse vectors, e.g. biases or skip layers.
    """
    def __init__(self, rc=1.0):
        self.rc = rc

    def __call__(self, x):
        return self.rc * torch.norm(x, p=1).item()

    def prox(self, x, delta=1.0):
        return torch.sign(x) * torch.clamp(torch.abs(x) - (delta * self.rc), min=0)

    def sub_grad(self, v):
        return self.rc * torch.sign(v)
