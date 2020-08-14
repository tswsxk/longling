# coding: utf-8
# create by tongshiwei on 2018/7/14

from .ValueMonitor import ValueMonitor, EMAValue

__all__ = ["LossMonitor", "MovingLoss"]


class LossMonitor(ValueMonitor):
    @property
    def losses(self):
        return self._values


class MovingLoss(EMAValue, LossMonitor):
    """
    Examples
    --------
    >>> lm = MovingLoss(["l2"])
    >>> lm.losses
    {'l2': nan}
    >>> lm("l2", 100)
    >>> lm("l2", 1)
    >>> lm["l2"]
    99.01
    """
    pass
