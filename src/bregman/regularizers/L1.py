import torch


class L1:
    def __init__(self, lamda=1.0):
        self.lamda = lamda

    def __call__(self, x):
        return self.lamda * torch.norm(x, p=1).item()

    def prox(self, x, delta=1.0):
        return torch.sign(x) * torch.clamp(torch.abs(x) - (delta * self.lamda), min=0)

    def sub_grad(self, v):
        return self.lamda * torch.sign(v)


class L1_pos:
    def __init__(self, lamda=1.0):
        self.lamda = lamda

    def __call__(self, x):
        return self.lamda * torch.norm(x, p=1).item()

    def prox(self, x, delta=1.0):
        return torch.clamp(
            torch.sign(x) * torch.clamp(torch.abs(x) - (delta * self.lamda), min=0),
            min=0,
        )

    def sub_grad(self, v):
        return self.lamda * torch.sign(v)
