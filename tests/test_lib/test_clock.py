# coding: utf-8
# 2019/12/3 @ tongshiwei

from longling import config_logging
from longling import Clock, print_time


def test_print_time(logger):
    with print_time("test", logger=logger):
        for _ in range(10):
            pass
        pass


def test_clock():
    clock = Clock()

    clock.start()

    for _ in range(100):
        pass

    clock.end(wall=True)
    clock.end(wall=False)

    print(clock.process_time)
    print(clock.wall_time)

    store_dict = {}
    with Clock(store_dict, config_logging(logger="clock")):
        for _ in range(100):
            pass

    assert "wall_time" in store_dict
    assert "process_time" in store_dict

    with Clock(store_dict, config_logging(logger="clock"), tips="testing clock"):
        for _ in range(100):
            pass
