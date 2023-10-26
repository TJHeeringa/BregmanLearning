import torch
from typing import Union, List, Callable, Optional


class LambdaScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, rc_lambda: Union[List[Callable], Callable]):
        self.optimizer = optimizer

        if not isinstance(rc_lambda, list) and not isinstance(rc_lambda, tuple):
            self.rc_lambdas = [rc_lambda] * len(optimizer.param_groups)
        else:
            if len(rc_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(rc_lambda)))
            self.rc_lambdas = list(rc_lambda)

        for group in optimizer.param_groups:
            group.setdefault('initial_rc', group['reg'].rc)

        self.base_rcs = [group['initial_rc'] for group in optimizer.param_groups]

    def step(self, epoch: Optional[int] = None):
        pass
