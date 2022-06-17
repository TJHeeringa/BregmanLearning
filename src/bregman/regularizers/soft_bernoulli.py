import torch


class SoftBernoulli:
    def __init__(self, lamda=1.0):
        self.lamda = lamda

    def prox(self, x, delta=1.0):
        return torch.sign(x) * torch.max(
            torch.clamp(torch.abs(x) - (delta * self.lamda), min=0),
            torch.bernoulli(0.01 * torch.ones_like(x)),
        )

    def sub_grad(self, v):
        return self.lamda * torch.sign(v)
