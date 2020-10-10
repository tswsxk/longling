# coding: utf-8
# 2019/11/28 @ tongshiwei


import math

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
    """

    def __init__(self, value_function_names: (list, dict), *args, **kwargs):
        self._values = {name: NAN for name in value_function_names}

    def __str__(self):
        return str(self._values)

    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __getitem__(self, item):
        return self._values[item]

    def update(self, *args, **kwargs):
        raise NotImplementedError

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
    def value(self):
        return self._values


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
    """

    def __init__(self, value_function_names: (list, dict), smoothing_constant=0.01):
        """
        Parameters
        ----------
        value_function_names
        smoothing_constant: float, optional
            Smoothing factor in range [0, 1], [default: 0.01].
            Increase to give more weight to recent values.
        """
        super(EMAValue, self).__init__(value_function_names)
        self.smoothing_constant = smoothing_constant

    def update(self, name: str, value: (float, int)):
        """
        Parameters
        ----------
        name: str
            The name of the value to be updated
        value  : int or float
            New value to include in EMA.
        """
        self._values[name] = (
            value if math.isnan(self._values[name])
            else (1 - self.smoothing_constant) * self._values[
                name] + self.smoothing_constant * value
        )
