# coding: utf-8
# 2020/4/19 @ tongshiwei

import pytest
from longling.ML.toolkit.monitor.ProgressMonitor import ConsoleProgressMonitor, ConsoleProgressMonitorPlayer


def test_monitor_player():
    values = {
        "Loss": {"l2": 0}
    }
    player = ConsoleProgressMonitorPlayer({"Loss": ["l2"]}, values=values)
    with player.watching():
        for i in range(100):
            player(i, Loss={"l2": 1})
    print(player.time)

    values = {
        "Loss": {"l2": 0}
    }
    player = ConsoleProgressMonitorPlayer({"Loss": ["l2"]}, values=values)
    with player.watching():
        for i in range(100):
            player(i)
    print(player.time)

    values = {
        "Loss": {"l2": 0}
    }
    player = ConsoleProgressMonitorPlayer({"Loss": ["l2"]}, values=values)
    with player.watching():
        for i in range(100):
            player(i, Eval={"l1": -1})
    print(player.time)


@pytest.mark.parametrize("player_type", ["default", "episode"])
def test_progress_monitor(player_type):
    values = {
        "Loss": {"l2": 0}
    }
    cp = ConsoleProgressMonitor({"Loss": ["l2"]}, values=values, player_type=player_type)
    for i in cp(range(100)):
        values["Loss"]["l2"] = i

    print(cp.iteration_time)


def test_epoch_progress_monitor():
    def iter_fn(num):
        for i in range(num):
            yield i

    values = {
        "Loss": {"l2": 0}
    }
    cp = ConsoleProgressMonitor({"Loss": ["l2"]}, values=values, player_type="epoch", total_epoch=2)

    for e in range(2):
        for i in cp(iter_fn(100), e + 1):
            values["Loss"]["l2"] = i

    print(cp.iteration_time)

    cp = ConsoleProgressMonitor({"Loss": ["l2"]}, values=values, player_type="epoch", total_epoch=2, silent=True)

    for e in range(2):
        for i in cp(iter_fn(100), e + 1):
            values["Loss"]["l2"] = i

    print(cp.iteration_time)
