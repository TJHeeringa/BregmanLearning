import torch


class Null:
    def __call__(self, x):
        return 0

    def prox(self, x, delta=1.0):
        return x

    def sub_grad(self, v):
        return torch.zeros_like(v)
