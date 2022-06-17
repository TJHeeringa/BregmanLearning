import torch


class L1_L2:
    def __init__(self, lamda=1.0):
        self.lamda = lamda

    def __call__(self, x):
        return self.lamda * torch.sqrt(x.shape[-1]) * torch.norm(torch.norm(x, p=2, dim=1), p=1).item()

    def prox(self, x, delta=1.0):
        thresh = delta * self.lamda
        thresh *= torch.sqrt(x.shape[-1])

        ret = torch.clone(x)
        nx = torch.norm(x, p=2, dim=1).view(x.shape[0], 1)

        ind = torch.where((nx != 0))[0]

        ret[ind] = x[ind] * torch.clamp(1 - torch.clamp(thresh / nx[ind], max=1), min=0)
        return ret

    def sub_grad(self, x):
        thresh = self.lamda * torch.sqrt(x.shape[-1])
        #
        nx = torch.norm(x, p=2, dim=1).view(x.shape[0], 1)
        ind = torch.where((nx != 0))[0]
        ret = torch.clone(x)
        ret[ind] = x[ind] / nx[ind]
        return thresh * ret
