import torch


class SoftBernoulli:
    def __init__(self, rc=1.0):
        self.rc = rc

    def prox(self, x, delta=1.0):
        return torch.sign(x) * torch.max(
            torch.clamp(torch.abs(x) - (delta * self.rc), min=0),
            torch.bernoulli(0.01 * torch.ones_like(x)),
        )

    def sub_grad(self, v):
        return self.rc * torch.sign(v)
