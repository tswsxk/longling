# coding: utf-8
# 2019/12/24 @ tongshiwei


class IncrementEstimator(object):
    def __init__(self, step, init_value=None):
        self._value = init_value
        self._step = step

    def __call__(self, value):
        self._value = self._value + self._step * (value - self._value) if self._value else value
        return self.value

    @property
    def value(self):
        return self._value
