import torch


class ProxOptimizer(torch.optim.Optimizer):
    # TODO: decide to keep or ditch
    def initialize_sub_grad(self, p, reg, delta):
        p_init = p.data.clone()
        return 1 / delta * p_init + reg.sub_grad(p_init)

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
