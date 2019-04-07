# coding: utf-8
# create by tongshiwei on 2018/7/14

import math

try:
    NAN = math.nan
except (AttributeError, ImportError):
    NAN = float('nan')

__all__ = ["MovingLoss"]


class LossMonitor(object):
    def __init__(self, loss_function_names, *args, **kwargs):
        self.losses = {name: NAN for name in loss_function_names}

    def __str__(self):
        return str(self.losses)

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        for name in self.losses:
            self.losses[name] = NAN

    def values(self):
        return self.losses.values()

    def items(self):
        return self.losses.items()

    def keys(self):
        raise self.losses.keys()


class MovingLoss(LossMonitor):
    def __init__(self, loss_function_names, smoothing_constant=0.01):
        super(MovingLoss, self).__init__(loss_function_names)
        self.smoothing_constant = smoothing_constant

    def update(self, name, loss_value):
        self.losses[name] = (
            loss_value if math.isnan(self.losses[name])
            else (1 - self.smoothing_constant) * self.losses[
                name] + self.smoothing_constant * loss_value
        )
