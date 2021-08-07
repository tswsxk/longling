# coding: utf-8
# 2019/11/28 @ tongshiwei

import warnings
import functools
import math
from collections import Iterable

try:
    NAN = math.nan
except (AttributeError, ImportError):  # pragma: no cover
    NAN = float('nan')

__all__ = ["ValueMonitor", "EMAValue", "as_tmt_value"]


class BaseTMTValue(object):
    """Basic Template is Module & module is Template (TMT) value"""

    def __init__(self):
        self.pipe = []

    def pipe_put(self, loss_value, *args, **kwargs):
        for _pipe in self.pipe:
            _pipe(loss_value, *args, **kwargs)


class TMTValue(BaseTMTValue):
    def __init__(self, value_obj, transform=lambda x: x):
        super(TMTValue, self).__init__()
        self._value_obj = value_obj
        self._transform = transform

    def __call__(self, *args, **kwargs):
        value = self._value_obj(*args, **kwargs)
        super(TMTValue, self).pipe_put(self._transform(value))
        return value


def as_tmt_value(value_obj, transform=lambda x: x):
    """

    Parameters
    ----------
    value_obj
    transform

    Returns
    -------

    Examples
    --------
    >>> def loss_f(a):
    ...     return a
    >>> loss_f(10)
    10
    >>> tmt_loss_f = as_tmt_value(loss_f)
    >>> tmt_loss_f(10)
    10
    >>> @as_tmt_value
    ... def loss_f2(a):
    ...     return a
    >>> loss_f2(10)
    10
    """
    return TMTValue(value_obj, transform)


# def tmt_value(transform=lambda x: x):
#     def _tmt_value(class_type):
#         class _TMTValue(class_type, BaseTMTValue):
#             @functools.wraps(class_type.__call__)
#             def __call__(self, *args, **kwargs):
#                 value = class_type.__call__(self, *args, **kwargs)
#                 BaseTMTValue.pipe_put(self, self.transform(value))
#                 return value
#
#             @staticmethod
#             def transform(loss):
#                 return transform(loss)
#
#         return _TMTValue
#
#     return _tmt_value


class ValueMonitor(object):
    """
    记录损失函数并显示

    几个重要的函数：
    * update: 定义损失函数的如何累加更新

    Examples
    --------
    >>> vm = ValueMonitor(["m1", "m2"])
    >>> vm.value
    {'m1': nan, 'm2': nan}
    >>> vm.keys()
    dict_keys(['m1', 'm2'])
    >>> vm.values()
    dict_values([nan, nan])
    >>> vm.items()
    dict_items([('m1', nan), ('m2', nan)])
    >>> "m1" in vm
    True
    >>> vm.monitor_off("m1")
    >>> "m1" in vm
    False
    >>> vm.monitor_on("m1")
    >>> "m1" in vm
    True
    >>> def m1_func():
    ...     return 1
    >>> def m2_func():
    ...     return 2
    >>> vm = ValueMonitor({"m1": m1_func, "m2": m2_func}, auto="ignore")
    >>> vm.value
    {'m1': nan, 'm2': nan}
    >>> vm["m1"]
    nan
    """

    def __init__(self, value_function_names: (list, dict), digits=6, auto=None, *args, **kwargs):
        """

        Parameters
        ----------
        value_function_names
        digits
        auto: str, None or bool
        args
        kwargs
        """
        self._values = {name: NAN for name in value_function_names}
        self._funcs = value_function_names if isinstance(value_function_names, dict) else {}
        self._auto = auto
        if self._auto is not False:
            for name, func in self._funcs.items():
                self.link(name, func)
        self._digits = digits

    def __str__(self):
        return str(self._values)

    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __getitem__(self, item):
        return self._values[item]

    def __contains__(self, name):
        return name in self._values

    def monitor_on(self, name, func=None):
        self._values[name] = NAN
        if func is not None:
            self._funcs[name] = func

    def monitor_off(self, name):
        del self._values[name]
        if name in self._funcs:
            del self._funcs[name]

    def get_update_value(self, name, value):
        raise NotImplementedError

    def update(self, name, value):
        self._values[name] = self.round(self.get_update_value(name, value))

    def reset(self):
        for name in self._values:
            self._values[name] = NAN

    def values(self):
        return self._values.values()

    def items(self):
        return self._values.items()

    def keys(self):
        return self._values.keys()

    @property
    def func(self):
        return self._funcs

    @property
    def value(self):
        return self._values

    def round(self, value):
        return round(value, self._digits) if self._digits else value

    def pipe_get(self, name, value):
        if name in self._values:
            self.update(name, value)

    def link(self, name, obj):
        if hasattr(obj, "pipe"):
            obj.pipe.append(functools.partial(self.pipe_get, name))
        else:
            hint = "invalid link operation, for %s has no pipe attribute, suggest to use TMTValue" % obj
            if self._auto is True:
                raise AttributeError(hint)
            elif self._auto is None or self._auto == "warn":
                warnings.warn(hint)


class EMAValue(ValueMonitor):
    """
    Exponential moving average: smoothing to give progressively lower
    weights to older values.

    .. math ::
        losses[name] = (1 - c) \\times previous\\_value + c \\times loss\\_value

    >>> ema = EMAValue(["l2"])
    >>> ema["l2"]
    nan
    >>> ema("l2", 100)
    >>> ema("l2", 1)
    >>> ema["l2"]
    90.1
    >>> list(ema.values())
    [90.1]
    >>> list(ema.keys())
    ['l2']
    >>> list(ema.items())
    [('l2', 90.1)]
    >>> ema.reset()
    >>> ema["l2"]
    nan
    >>> ema = EMAValue(["l1", "l2"])
    >>> ema["l2"], ema["l1"]
    (nan, nan)
    >>> ema.updates({"l1": 1, "l2": 10})
    >>> ema.updates({"l1": 10, "l2": 100})
    >>> ema["l1"]
    1.9
    >>> ema["l2"]
    19.0
    >>> ema = EMAValue(["l1"], smoothing_constant=0.0)
    >>> ema["l1"]
    nan
    >>> ema.updates({"l1": 1})
    >>> ema.updates({"l1": 10})
    >>> ema["l1"]
    1.0
    >>> ema = EMAValue(["l1"], smoothing_constant=1.0)
    >>> ema.updates({"l1": 1})
    >>> ema.updates({"l1": 10})
    >>> ema["l1"]
    10.0
    >>> @as_tmt_value
    ... def mse_loss(a):
    ...     return a ** 2
    >>> ema = EMAValue({"mse": mse_loss})
    >>> ema["mse"]
    nan
    >>> mse_loss(1)
    1
    >>> ema["mse"]
    1
    >>> mse_loss(10)
    100
    >>> ema["mse"]
    10.9
    >>> ema = EMAValue({"mse": mse_loss})
    >>> mse_loss(1)
    1
    >>> ema["mse"]
    1
    >>> ema.monitor_off("mse")
    >>> ema.func
    {}
    >>> mse_loss(10)
    100
    >>> "mse" not in ema
    True
    >>> ema.monitor_on("mse", mse_loss)
    >>> mse_loss(10)
    100
    >>> ema["mse"]
    100
    """

    def __init__(self, value_function_names: (list, dict), smoothing_constant=0.1, *args, **kwargs):
        """
        Parameters
        ----------
        value_function_names
        smoothing_constant: float, optional
            Smoothing factor in range [0, 1], [default: 0.01].
            Increase to give more weight to recent values.
        """
        super(EMAValue, self).__init__(value_function_names, *args, **kwargs)
        self.smoothing_constant = smoothing_constant

    def get_update_value(self, name: str, value: (float, int)):
        """
        Parameters
        ----------
        name: str
            The name of the value to be updated
        value  : int or float
            New value to include in EMA.
        """
        return (
            value if math.isnan(self._values[name])
            else (1 - self.smoothing_constant) * self._values[
                name] + self.smoothing_constant * value
        )

    def updates(self, pairs: (dict, Iterable)):
        pairs = pairs.items() if isinstance(pairs, dict) else pairs
        for name, value in pairs:
            self.update(name, value)
