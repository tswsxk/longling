# coding: utf-8
# create by tongshiwei on 2018/7/14

# from .ValueMonitor import ValueMonitor, EMAValue, as_tmt_value
from longling import as_list
from longling.ML.toolkit.monitor.ValueMonitor import ValueMonitor, EMAValue, as_tmt_value
__all__ = ["LossMonitor", "MovingLoss", "as_tmt_loss", "loss_dict2tmt_loss"]


# def tmt_loss(loss2value=lambda x: x):
#     return tmt_value(transform=loss2value)


def as_tmt_loss(loss_obj, loss2value=lambda x: x):
    """

    Parameters
    ----------
    loss_obj
    loss2value

    Returns
    -------

    Examples
    --------
    >>> @as_tmt_loss
    ... def mse(v):
    ...     return v ** 2
    >>> mse(2)
    4
    """
    return as_tmt_value(loss_obj, loss2value)


def loss_dict2tmt_loss(loss_dict, loss2value=lambda x: x, exclude=None, include=None, as_loss=as_tmt_loss):
    """

    Parameters
    ----------
    loss_dict
    loss2value
    exclude
    include
    as_loss

    Returns
    -------

    Examples
    --------
    >>> def mse(v):
    ...     return v ** 2
    >>> losses = loss_dict2tmt_loss({"mse": mse, "rmse": lambda x: x})
    >>> losses.keys()
    dict_keys(['mse', 'rmse'])
    >>> ema = EMAValue(losses)
    >>> losses["mse"](2)
    4
    >>> losses["rmse"](2)
    2
    >>> ema.items()
    dict_items([('mse', 4), ('rmse', 2)])
    >>> losses = loss_dict2tmt_loss({"mse": mse, "rmse": lambda x: x}, include="mse")
    >>> losses.keys()
    dict_keys(['mse', 'rmse'])
    >>> ema = EMAValue(losses, auto="ignore")
    >>> losses["mse"](2)
    4
    >>> losses["rmse"](2)
    2
    >>> ema.items()
    dict_items([('mse', 4), ('rmse', nan)])
    >>> losses = loss_dict2tmt_loss({"mse": mse, "rmse": lambda x: x}, exclude="mse")
    >>> losses.keys()
    dict_keys(['mse', 'rmse'])
    >>> ema = EMAValue(losses, auto="ignore")
    >>> losses["mse"](2)
    4
    >>> losses["rmse"](2)
    2
    >>> ema.items()
    dict_items([('mse', nan), ('rmse', 2)])
    """
    exclude = set() if exclude is None else set(as_list(exclude))
    if include is not None:
        include = set(as_list(include))
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
