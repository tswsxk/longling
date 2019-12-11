# coding: utf-8
# create by tongshiwei on 2019-9-1
__all__ = ["IterableMonitor", "MonitorPlayer", "ProgressMonitor"]

"""
进度监视器，帮助用户知晓当前运行进度 

一个简单的示例如下 ::

    class DemoMonitor(ProgressMonitor):
        def __call__(self, iterator):
            return IterableMonitor(
                iterator,
                self.player, self.player.set_length
            )


    progress_monitor = DemoMonitor(MonitorPlayer())

    for _ in range(5):
        for _ in progress_monitor(range(10000)):
            pass
        print()
"""

from collections import Iterable

try:
    from .stream import flush_print
except (ImportError, SystemError):
    from stream import flush_print


def pass_function(*args, **kwargs):
    pass


class IterableMonitor(Iterable):
    """
    迭代监控器

    Parameters
    ----------
    iterator:
        待迭代数据
    call_in_iter:
        每次迭代中的回调函数（例如：打印进度等）
    call_after_iter:
        每轮迭代后的回调函数（所有数据遍历一遍后）
    length:
        数据总长（有多少条数据）
    """

    def __init__(self,
                 iterator: (Iterable, list, tuple, dict),
                 call_in_iter=pass_function, call_after_iter=pass_function,
                 length: (int, None) = None,
                 ):
        self.iterator = iter(iterator)
        self.call_in_iter = call_in_iter
        self.call_after_iter = call_after_iter

        self._count = 0
        try:
            if length is not None:
                self._length = length
            else:
                self._length = len(iterator)
        except TypeError:
            self._length = None

    def set_length(self, length):
        self._length = length

    def reset(self):
        self._count = 0
        self._length = None

    def __len__(self):
        return self._length

    def __next__(self):
        try:
            res = self.iterator.__next__()
            self._count += 1
            self.call_in_iter(self._count)
            return res
        except StopIteration:
            self._length = self._count
            self._count = 0
            self.call_after_iter(self._length)
            raise StopIteration

    def __iter__(self):
        return self


class MonitorPlayer(object):
    """监控器显示器"""

    def __init__(self):
        self._count = 0
        self._length = None

    def reset(self):
        self._count = 0
        self._length = None

    def __call__(self, _count):
        self._count = _count
        flush_print("%s|%s" % (self._count, self._length))

    def set_length(self, _length):
        self._length = _length


class ProgressMonitor(object):
    def __init__(self, player):
        self.player = player

    def __call__(self, iterator):
        raise NotImplementedError


if __name__ == '__main__':
    class DemoMonitor(ProgressMonitor):
        def __call__(self, iterator):
            return IterableMonitor(
                iterator,
                self.player, self.player.set_length
            )


    progress_monitor = DemoMonitor(MonitorPlayer())

    for _ in range(5):
        for _ in progress_monitor(range(10000)):
            pass
        print()
