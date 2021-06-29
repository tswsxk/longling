# coding: utf-8
# 2021/6/28 @ tongshiwei
from ..toolkit import result_format
from collections import OrderedDict

__all__ = ["POrderedDict"]


class POrderedDict(OrderedDict):
    """
    Examples
    --------
    >>> _d = {"a": 1, "b": {0: 2, 1: 3}, "c": {0: 1, 1: 5}}
    >>> _d
    {'a': 1, 'b': {0: 2, 1: 3}, 'c': {0: 1, 1: 5}}
    >>> POrderedDict(_d)
       0  1
    b  2  3
    c  1  5
    a: 1
    """

    def __repr__(self):
        return result_format(self)
