# coding: utf-8
# create by tongshiwei on 2018/7/14

from .ValueMonitor import ValueMonitor, MovingValue

__all__ = ["LossMonitor", "MovingLoss"]


class LossMonitor(ValueMonitor):
    @property
    def losses(self):
        return self._values


class MovingLoss(MovingValue, LossMonitor):
    pass
