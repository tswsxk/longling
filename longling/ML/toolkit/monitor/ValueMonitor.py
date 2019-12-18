# coding: utf-8
# 2019/11/28 @ tongshiwei


import math

try:
    NAN = math.nan
except (AttributeError, ImportError):
    NAN = float('nan')

__all__ = ["ValueMonitor", "MovingValue"]


class ValueMonitor(object):
    """
    记录损失函数并显示

    几个重要的函数：
    * update: 定义损失函数的如何累加更新

    """

    def __init__(self, value_function_names, *args, **kwargs):
        self._values = {name: NAN for name in value_function_names}

    def __str__(self):
        return str(self._values)

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
        raise self._values.keys()


class MovingValue(ValueMonitor):
    def __init__(self, value_function_names, smoothing_constant=0.01):
        super(MovingValue, self).__init__(value_function_names)
        self.smoothing_constant = smoothing_constant

    def update(self, name, value):
        """
        ..math:
        losses[name] = 1 - c \times loss_value + c \times loss_value
        """
        self._values[name] = (
            value if math.isnan(self._values[name])
            else (1 - self.smoothing_constant) * self._values[
                name] + self.smoothing_constant * value
        )
