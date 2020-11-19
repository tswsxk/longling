# coding: utf-8
# create by tongshiwei on 2018/7/14

from .ValueMonitor import ValueMonitor, EMAValue, tmt_value, as_tmt_value

__all__ = ["LossMonitor", "MovingLoss", "tmt_loss", "as_tmt_loss"]


def tmt_loss(loss2value=lambda x: x):
    return tmt_value(transform=loss2value)


def as_tmt_loss(loss_obj, loss2value=lambda x: x):
    return as_tmt_value(loss_obj, loss2value)


class LossMonitor(ValueMonitor):
    @property
    def losses(self):
        return self.value


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
