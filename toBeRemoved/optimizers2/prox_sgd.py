import torch
from bregman import regularizers


class ProxSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, reg=regularizers.Null()):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")

        defaults = dict(lr=lr, reg=reg)
        super(ProxSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            # define regularizer for this group
            reg = group["reg"]
            step_size = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                # get grad and state
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0

                # -------------------------------------------------------------
                # update scheme
                # -------------------------------------------------------------
                # gradient steps
                p.data.add_(-step_size * grad)
                # proximal step
                p.data = reg.prox(p.data, step_size)

    @torch.no_grad()
    def evaluate_reg(self):
        reg_vals = []
        for group in self.param_groups:
            group_reg_val = 0.0
            # define regularizer for this group
            reg = group["reg"]

            # evaluate the reguarizer for each parametr in group
            for p in group["params"]:
                group_reg_val += reg(p)

            # append the group reg val
            reg_vals.append(group_reg_val)

        return reg_vals
