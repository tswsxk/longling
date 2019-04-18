# coding: utf-8
# create by tongshiwei on 2019/4/17

import functools
from longling import ML

__all__ = ["part_dependency", "__dependency__"]

part_dependency = {
    "ML": ML.__dependency__,
}

__dependency__ = list(set(functools.reduce(
    lambda x, y: x + y,
    part_dependency.values()
)))

if __name__ == '__main__':
    print("\n".join(__dependency__))
