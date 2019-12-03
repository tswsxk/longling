# coding: utf-8
# 2019/12/3 @ tongshiwei

from longling import Clock, print_time


def test_print_time(logger):
    with print_time("test", logger=logger):
        a = 1 + 1
    assert True
