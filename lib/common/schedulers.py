# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

# Copyright (c) Facebook, Inc. and its affiliates.


from bisect import bisect_right
from lib.common.registry import registry
from torch.optim.lr_scheduler import LambdaLR


@registry.register_scheduler("pythia")
class PythiaScheduler(LambdaLR):
    def __init__(self, optimizer, *args, **kwargs):
        from lib.utils.general import lr_lambda_update

        self._lambda_func = lr_lambda_update
        self._global_config = registry.get("config")

        super().__init__(optimizer, self.lr_lambda, *args, **kwargs)

    def lr_lambda(self, step):
        return self._lambda_func(step, self._global_config)


@registry.register_scheduler("multi_step")
class MultiStepScheduler(PythiaScheduler):
    def __init__(self, optimizer, *args, **kwargs):
        self.use_warmup = kwargs["use_warmup"]
        self.lr_steps = kwargs["lr_steps"]
        self.lr_ratio = kwargs["lr_ratio"]
        self.warmup_iterations = kwargs["warmup_iterations"] if self.use_warmup else 0
        self.warmup_factor = kwargs["warmup_factor"]
        assert self.warmup_iterations < self.lr_steps[0]
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iterations and self.use_warmup is True:
            alpha = float(self.last_epoch) / float(self.warmup_iterations)
            lr_ratio = self.warmup_factor * (1.0 - alpha) + alpha

            return [base_lr * lr_ratio for base_lr in self.base_lrs]
        else:
            return [
                base_lr * self.lr_ratio ** bisect_right(self.lr_steps, self.last_epoch)
                for base_lr in self.base_lrs
            ]