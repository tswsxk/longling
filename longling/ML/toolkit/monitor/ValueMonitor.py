# coding: utf-8
# 2019/11/28 @ tongshiwei

import math
from collections import Iterable

try:
    NAN = math.nan
except (AttributeError, ImportError):  # pragma: no cover
    NAN = float('nan')

__all__ = ["ValueMonitor", "EMAValue"]


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
    >>> "m1" in vm
    True
    >>> vm.monitor_off("m1")
    >>> "m1" in vm
    False
    >>> vm.monitor_on("m1")
    >>> "m1" in vm
    True
    """

    def __init__(self, value_function_names: (list, dict), digits=6, *args, **kwargs):
        self._values = {name: NAN for name in value_function_names}
        self._funcs = value_function_names if isinstance(value_function_names, dict) else {}
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
        if self._digits:
            return {key: round(value, self._digits) for key, value in self._values.items()}
        else:
            return self._values

    def round(self, value):
        return round(value, self._digits) if self._digits else value


class EMAValue(ValueMonitor):
    """
    Exponential moving average: smoothing to give progressively lower
    weights to older values.

    ..math:
        losses[name] = 1 - c \times previous_value + c \times loss_value

    >>> ema = EMAValue(["l2"])
    >>> ema["l2"]
    nan
    >>> ema("l2", 100)
    >>> ema("l2", 1)
    >>> ema["l2"]
    99.01
    >>> list(ema.values())
    [99.01]
    >>> list(ema.keys())
    ['l2']
    >>> list(ema.items())
    [('l2', 99.01)]
    >>> ema.reset()
    >>> ema["l2"]
    nan
    >>> ema = EMAValue(["l1", "l2"])
    >>> ema["l2"], ema["l1"]
    (nan, nan)
    >>> ema.updates({"l1": 1, "l2": 10})
    >>> ema.updates({"l1": 10, "l2": 100})
    >>> ema["l1"]
    1.09
    >>> ema["l2"]
    10.9
    """

    def __init__(self, value_function_names: (list, dict), smoothing_constant=0.01, *args, **kwargs):
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
