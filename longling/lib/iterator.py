# coding: utf-8
# 2020/1/13 @ tongshiwei
import json
import os
import warnings
import functools
import queue
import threading
import multiprocessing as mp
from tqdm import tqdm

from longling import wf_open, loading

__all__ = ["BaseIter", "MemoryIter", "LoopIter", "AsyncLoopIter", "AsyncIter", "CacheAsyncLoopIter", "iterwrap"]


class Register(dict):  # pragma: no cover
    def add(self, cls):
        if cls.__name__ in self:
            warnings.warn("key %s has already existed, which will be overridden" % cls.__name__)
        self[cls.__name__] = cls
        return cls


register = Register()


@register.add
class BaseIter(object):
    """
    迭代器

    Notes
    ------

    * 如果 src 是一个迭代器实例，那么在一轮迭代之后，迭代器里的内容就被迭代完了，将无法重启。
    * 如果想使得迭代器可以一直被循环迭代，那么 src 应当是迭代器实例的生成函数, 同时在每次循环结束后，调用reset()
    * 如果 src 没有 __length__，那么在第一次迭代结束前，无法对 BaseIter 的实例调用 len() 函数

    Examples
    --------

    .. code-block:: python

        # 单次迭代后穷尽内容
        with open("demo.txt") as f:
            bi = BaseIter(f)
            for line in bi:
                pass

        # 可多次迭代
        def open_file():
            with open("demo.txt") as f:
                for line in f:
                    yield line

        bi = BaseIter(open_file)
        for _ in range(5):
            for line in bi:
                pass
            bi.reset()


        # 简化的可多次迭代的写法
        @BaseIter.wrap
        def open_file():
            with open("demo.txt") as f:
                for line in f:
                    yield line

        bi = open_file()
        for _ in range(5):
            for line in bi:
                pass
            bi.reset()
    """

    def __init__(self, src, length=None, *args, **kwargs):
        self._reset = src
        self._data = None
        self._length = length
        self._count = 0
        self.init()

    def __next__(self):
        try:
            self._count += 1
            return next(self._data)
        except StopIteration:
            self._count -= 1
            self._set_length()
            raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        if self._length is None:
            raise TypeError("length is unknown")
        return self._length

    def init(self):
        self.reset()

    def reset(self):
        _data = self._reset() if callable(self._reset) else self._reset
        try:
            self._length = len(_data)
        except TypeError:
            self._set_length()
        self._data = iter(_data)

    def _set_length(self):
        if self._length is None and self._count > 0:
            self._length = self._count

    @classmethod
    def wrap(cls, f):
        @functools.wraps(f)
        def _f(*args, **kwargs):
            return cls(functools.partial(f, *args, **kwargs))

        return _f


@register.add
class MemoryIter(BaseIter):
    """
    内存迭代器

    会将所有迭代器内容装载入内存
    """

    def __init__(self, src, length=None, prefetch=False, *args, **kwargs):
        self._memory_data = []
        self._in_memory = True
        super(MemoryIter, self).__init__(src, length)
        self.prefetch = prefetch
        if self.prefetch:
            self._prefetch()

    def _prefetch(self):
        for _data in tqdm(self._data, "prefetching data"):
            self._count += 1
            self._memory_data.append(_data)
        self._set_length()
        self._in_memory = False
        self._data = iter(self._memory_data)

    def __next__(self):
        try:
            self._count += 1
            elem = next(self._data)
            if self._in_memory is True:
                self._memory_data.append(elem)
            return elem
        except StopIteration:
            self._count -= 1
            self.reset()
            self._in_memory = False
            raise StopIteration

    def reset(self):
        if not self._memory_data:
            super(MemoryIter, self).reset()
        else:
            self._data = iter(self._memory_data)
            self._set_length()


@register.add
class LoopIter(BaseIter):
    """
    循环迭代器

    每次迭代后会进行自动的 reset() 操作
    """

    def __init__(self, src, length=None, *args, **kwargs):
        super(LoopIter, self).__init__(src, length)

    def __next__(self):
        try:
            self._count += 1
            return next(self._data)
        except StopIteration:
            self._count -= 1
            self.reset()
            raise StopIteration


def produce(data, produce_queue):
    _stop = False
    try:
        for _data in data:
            produce_queue.put(_data)
        raise StopIteration

    except StopIteration as e:
        if not _stop:
            _stop = True
            produce_queue.put(e)

    except Exception as e:  # pragma: no cover
        if not _stop:
            _stop = True
            produce_queue.put(e)


@register.add
class AsyncLoopIter(LoopIter):
    """
    异步循环迭代器,适用于加载文件

    数据的读入和数据的使用迭代是异步的。reset() 之后会进行数据预取
    """

    def __init__(self, src, tank_size=8, timeout=None, level="t"):
        self.thread = None
        self._size = tank_size
        self.mode = self._mode_map(level)
        if self.mode == "t":
            self.queue_cls = queue.Queue
            self.thread_cls = threading.Thread
        elif self.mode == "p":
            self.queue_cls = mp.Queue
            self.thread_cls = mp.Process
        else:  # pragma: no cover
            raise TypeError("unknown mode: %s" % self.mode)
        self.queue = self.queue_cls(self._size)
        self._timeout = timeout
        super(AsyncLoopIter, self).__init__(src)

    @classmethod
    def _mode_map(cls, mode):
        map_dict = {
            "p": "p", "processing": "p", "multiprocessing": "p",
            "t": "t", "thread": "t", "threading": "t",
        }
        return map_dict[mode]

    def reset(self):
        super(AsyncLoopIter, self).reset()
        if self.thread is not None:
            self.thread.join()

        self.thread = self.thread_cls(
            target=produce,
            kwargs=dict(data=self._data, produce_queue=self.queue),
            daemon=True
        )
        self.thread.start()

    def __next__(self):
        if self.queue is not None:
            item = self.queue.get()
        else:  # pragma: no cover
            raise StopIteration
        if isinstance(item, Exception):
            if isinstance(item, StopIteration):
                self.reset()
                raise StopIteration
            else:  # pragma: no cover
                raise item
        else:
            self._count += 1
            return item


@register.add
class AsyncIter(AsyncLoopIter):
    """
    异步装载迭代器

    不会进行自动 reset()
    """

    def init(self):
        super(AsyncIter, self).reset()

    def reset(self):
        self.queue = queue.Queue(self._size)
        super(AsyncIter, self).reset()

    def __next__(self):
        if self.queue is not None:
            item = self.queue.get()
        else:
            raise StopIteration
        if isinstance(item, Exception):
            if isinstance(item, StopIteration):
                self.thread = None
                self.queue = None
                self._set_length()
                raise StopIteration
            else:  # pragma: no cover
                raise item
        else:
            self._count += 1
            return item


@register.add
class CacheAsyncLoopIter(AsyncLoopIter):
    """
    带缓冲池的异步迭代器，适用于带预处理的文件

    自动 reset(),
    同时针对 src 为 function 时可能存在的复杂预处理（即异步加载取数据操作比迭代输出数据操作时间长很多），
    将异步加载中处理的预处理数据放到指定的缓冲文件中
    """

    def __init__(self, src, cache_file, rerun=True, tank_size=8, timeout=None, level="t"):
        self.cache_file = cache_file
        self.cache_queue = None
        self.cache_thread = None

        if os.path.exists(self.cache_file) and rerun is False:
            # 从已有数据中进行装载
            def src():
                return loading(self.cache_file, "jsonl")

            self._cache_stop = True
        else:
            # 重新生成数据
            self.cache_queue = queue.Queue(tank_size)
            self.cache_thread = threading.Thread(target=self.cached, daemon=False)
            self.cache_thread.start()
            self._cache_stop = False
        super(CacheAsyncLoopIter, self).__init__(src, tank_size, timeout, level)

    def init(self):
        super(CacheAsyncLoopIter, self).reset()

    def reset(self):
        self._reset = lambda: loading(self.cache_file, "jsonl")
        if self.cache_thread is not None:
            self.cache_thread.join()
            self.cache_thread = None
            self.cache_queue = None
        super(CacheAsyncLoopIter, self).reset()

    def __next__(self):
        item = self.queue.get()
        if self.cache_queue is not None:
            self.cache_queue.put(item)
        if isinstance(item, Exception):
            if isinstance(item, StopIteration):
                self.reset()
                raise StopIteration
            else:  # pragma: no cover
                raise item
        else:
            self._count += 1
            return item

    def cached(self):
        assert self.cache_queue is not None

        with wf_open(self.cache_file, mode="w") as wf:
            while True:
                data = self.cache_queue.get()
                if isinstance(data, StopIteration):
                    break
                print(json.dumps(data), file=wf)


def iterwrap(itertype: str = "AsyncLoopIter", *args, **kwargs):
    """
    迭代器装饰器，适用于希望能重复使用一个迭代器的情况，能将迭代器生成函数转换为可以重复使用的函数。
    默认使用 AsyncLoopIter。

    Examples
    --------

    .. code-block:: python

        @iterwrap()
        def open_file():
            with open("demo.txt") as f:
                for line in f:
                    yield line

        data = open_file()
        for _ in range(5):
            for line in data:
                pass
    """
    if itertype not in register:
        raise TypeError("itertype %s is unknown, the available type are %s" % (itertype, ", ".join(register)))

    def _f1(f):
        @functools.wraps(f)
        def _f2(*fargs, **fkwargs):
            return register[itertype](functools.partial(f, *fargs, **fkwargs), *args, **kwargs)

        return _f2

    return _f1
