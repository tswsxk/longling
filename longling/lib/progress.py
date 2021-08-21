# coding: utf-8
# create by tongshiwei on 2019-9-1
__all__ = ["IterableMIcing", "MonitorPlayer", "ProgressMonitor", "AsyncMonitorPlayer"]

"""
进度监视器，帮助用户知晓当前运行进度，主要适配于机器学习中分 epoch，batch 的情况。

和 tqdm 针对单个迭代对象进行快速适配不同，
progress的目标是能将监视器不同功能部件模块化后再行组装，可以实现description的动态化，
给用户提供更大的便利性。

* MonitorPlayer 定义了如何显示进度和其它过程参数(better than tqdm, where only n is changed and description is fixed)
    * 在 __call__ 方法中定义如何显示
* 继承ProgressMonitor并传入必要参数进行实例化
    * 继承重写ProgressMonitor的__call__函数，用 IterableMIcing 包裹迭代器，这一步可以灵活定义迭代前后的操作
    * 需要在__init__的时候传入一个MonitorPlayer实例
* IterableMIcing 用来组装迭代器、监控器

一个简单的示例如下

.. code-block:: python

    class DemoMonitor(ProgressMonitor):
        def __call__(self, iterator):
            return IterableMIcing(
                iterator,
                self.player, self.player.set_length
            )

    progress_monitor = DemoMonitor(MonitorPlayer())

    for _ in range(5):
        for _ in progress_monitor(range(10000)):
            pass
        print()

cooperate with tqdm

.. code-block:: python

    from tqdm import tqdm

    class DemoTqdmMonitor(ProgressMonitor):
        def __call__(self, iterator, **kwargs):
            return tqdm(iterator, **kwargs)
"""

from typing import Iterable
from longling.lib.stream import flush_print
import threading
import queue


def pass_function(*args, **kwargs):
    pass


class IterableMIcing(Iterable):
    """
    将迭代器包装为监控器可以使用的迭代类：
    * 添加计数器 count, 每迭代一次，count + 1, 迭代结束时，可根据 count 得知数据总长
    * 每次 __iter__ 时会调用 call_in_iter 函数
    * 迭代结束时，会调用 call_after_iter

    Parameters
    ----------
    iterator:
        待迭代数据
    hook_in_iter:
        每次迭代中的回调函数（例如：打印进度等）,接受当前的 count 为输入
    hook_after_iter:
        每轮迭代后的回调函数（所有数据遍历一遍后），接受当前的 length 为输入
    length:
        数据总长（有多少条数据）

    >>> iterator = IterableMIcing(range(100))
    >>> for i in iterator:
    ...     pass
    >>> len(iterator)
    100
    >>> def iter_fn(num):
    ...     for i in range(num):
    ...         yield num
    >>> iterator = IterableMIcing(iter_fn(50))
    >>> for i in iterator:
    ...     pass
    >>> len(iterator)
    50
    """

    def __init__(self,
                 iterator: (Iterable, list, tuple, dict),
                 hook_in_iter=pass_function, hook_after_iter=pass_function,
                 length: (int, None) = None,
                 ):
        self.iterator = iter(iterator)
        self.call_in_iter = hook_in_iter
        self.call_after_iter = hook_after_iter

        self._count = 0
        try:
            if isinstance(length, int):
                self._length = length
            elif length is None:
                self._length = len(iterator)
            else:
                raise TypeError()
        except TypeError:
            self._length = None

    def set_length(self, length):
        self._length = length

    def reset(self, iterator):
        self.iterator = iter(iterator)
        self._count = 0
        self._length = None

    def __len__(self):
        """TypeError will be raised when _length is None."""
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


class AsyncMonitorPlayer(object):  # pragma: no cover
    """异步监控器显示器"""

    def __init__(self, cache_size=10000):
        import warnings
        warnings.warn("dev version, do not use")

        self._count = 0
        self._length = None
        self._size = cache_size
        self.thread = None
        self._display_cache = None
        self.reset()

    def display(self):
        while True:
            _count = self._display_cache.get()
            if isinstance(_count, StopIteration):
                return
            self._count = _count
            flush_print("%s|%s" % (self._count, self._length))

    def reset(self):
        self._count = 0
        self._length = None
        if self.thread is not None:
            self._display_cache.put(StopIteration())
            self.thread.join()
        self.thread = threading.Thread(target=self.display, daemon=True)
        self._display_cache = queue.Queue(self._size)
        self.thread.start()

    def __call__(self, _count):
        self._display_cache.put(_count)

    def set_length(self, _length):
        self._length = _length


class MonitorPlayer(object):
    """异步监控器显示器"""

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
    def __init__(self, player=None):
        self.player = player

    def __call__(self, iterator, *args, **kwargs):
        raise NotImplementedError
