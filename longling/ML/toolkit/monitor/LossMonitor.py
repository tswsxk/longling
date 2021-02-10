# coding: utf-8
# create by tongshiwei on 2018/7/14

from .ValueMonitor import ValueMonitor, EMAValue, tmt_value, as_tmt_value

__all__ = ["LossMonitor", "MovingLoss", "tmt_loss", "as_tmt_loss", "loss_dict2tmt_loss"]


def tmt_loss(loss2value=lambda x: x):
    return tmt_value(transform=loss2value)


def as_tmt_loss(loss_obj, loss2value=lambda x: x):
    return as_tmt_value(loss_obj, loss2value)


def loss_dict2tmt_loss(loss_dict, loss2value=lambda x: x, exclude=None, include=None, as_loss=as_tmt_loss):
    exclude = set() if exclude is None else set(exclude)
    if include is not None:
        include = set(include)
        return {
            name: as_loss(func, loss2value) if name in include else func for name, func in loss_dict.items()
        }
    return {
        name: as_loss(func, loss2value) if name not in exclude else func for name, func in loss_dict.items()
    }


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
    90.1
    """
    pass
