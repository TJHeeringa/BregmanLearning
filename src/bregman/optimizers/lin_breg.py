import torch
from bregman import regularizers


class LinBreg(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, reg=regularizers.Null(), delta=1.0, momentum=0.0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")

        defaults = dict(lr=lr, reg=reg, delta=delta, momentum=momentum)
        super(LinBreg, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            delta = group["delta"]
            # define regularizer for this group
            reg = group["reg"]
            step_size = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                # get grad and state
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # get prox
                    # initialize subgradients
                    state["sub_grad"] = self.initialize_sub_grad(p, reg, delta)
                    state["momentum_buffer"] = None
                # -------------------------------------------------------------
                # update scheme
                # -------------------------------------------------------------
                # get the current sub gradient
                sub_grad = state["sub_grad"]
                # update on the subgradient
                if momentum > 0.0:  # with momentum
                    mom_buff = state["momentum_buffer"]
                    if state["momentum_buffer"] is None:
                        mom_buff = torch.zeros_like(grad)

                    mom_buff.mul_(momentum)
                    mom_buff.add_((1 - momentum) * step_size * grad)
                    state["momentum_buffer"] = mom_buff
                    # update subgrad
                    sub_grad.add_(-mom_buff)

                else:  # no momentum
                    sub_grad.add_(-step_size * grad)
                # update step for parameters
                p.data = reg.prox(delta * sub_grad, delta)

    def initialize_sub_grad(self, p, reg, delta):
        p_init = p.data.clone()
        return 1 / delta * p_init + reg.sub_grad(p_init)

    @torch.no_grad()
    def evaluate_reg(self):
        reg_vals = []
        for group in self.param_groups:
            group_reg_val = 0.0
            delta = group["delta"]

            # define regularizer for this group
            reg = group["reg"]

            # evaluate the reguarizer for each parametr in group
            for p in group["params"]:
                group_reg_val += reg(p)

            # append the group reg val
            reg_vals.append(group_reg_val)

        return reg_vals
