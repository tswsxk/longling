# coding: utf-8
# 2019/12/11 @ tongshiwei

from longling.lib.progress import ProgressMonitor, IterableMonitor, MonitorPlayer


class DemoMonitor(ProgressMonitor):
    def __call__(self, iterator):
        return IterableMonitor(
            iterator,
            self.player, self.player.set_length
        )


def test_progress():
    progress_monitor = DemoMonitor(MonitorPlayer())

    for _ in range(5):
        for _ in progress_monitor(range(10000)):
            pass
        print()
