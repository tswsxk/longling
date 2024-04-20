# coding: utf-8
# 2019/12/11 @ tongshiwei

import pytest
from longling.utils.progress import ProgressMonitor, IterableMIcing, MonitorPlayer


class DemoMonitor(ProgressMonitor):
    def __call__(self, iterator):
        return IterableMIcing(
            iterator,
            self.player, self.player.set_length
        )


def test_progress():
    progress_monitor = DemoMonitor(MonitorPlayer())

    for _ in progress_monitor(range(10)):
        pass


def test_exception():
    for _ in IterableMIcing(range(10), length=0.6):
        pass

    for _ in IterableMIcing(range(10), length=10):
        pass

    im = IterableMIcing(range(10), length=5)
    im.set_length(10)
    assert len(im) == 10

    for _ in im:
        pass

    im.reset(range(10))

    with pytest.raises(TypeError):
        len(im)

    mp = MonitorPlayer()
    mp.reset()

    pm = ProgressMonitor(mp)

    with pytest.raises(NotImplementedError):
        pm(range(10))
