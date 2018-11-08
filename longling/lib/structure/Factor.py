# coding:utf-8
# created by tongshiwei on 2018/11/8
import numpy as np


class Factor(object):
    def __init__(self, initial_value, end_value, change_rate=0, warmup_steps=0):
        self.cnt = 0
        self._initial_value = initial_value
        self._end_value = end_value
        self._sign = int(np.sign(initial_value - end_value))
        self._value = initial_value
        self.warmup_steps = warmup_steps
        self._change_rate = - self._sign * change_rate

    @property
    def value(self):
        if self._sign == 0:
            return self._value
        else:
            if self.cnt >= self.warmup_steps and self._sign * (self._value - self._end_value) > 0:
                self._value += self._change_rate
            else:
                self.cnt += 1

    def reset(self):
        self._value = self._initial_value
        self.cnt = 0


def linearly_decay(initial_rate, min_rate, period, warmup=0):
    assert initial_rate > min_rate
    return initial_rate, min_rate, (initial_rate - min_rate) / (period - warmup)
