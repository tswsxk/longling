# coding: utf-8
# create by tongshiwei on 2019/4/17

import functools
from longling import ML

part_dependency = {
    "ML": ML.__dependency__,
}

__dependency__ = list(set(functools.reduce(
    lambda x, y: x + y,
    part_dependency.values()
)))
