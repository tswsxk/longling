# coding: utf-8
# 2020/1/13 @ tongshiwei
import json
import os
import uuid
import tempfile
import warnings
import functools
import queue
import threading
import logging
from longling import wf_open, path_append, loading

__all__ = ["BaseIter", "LoopIter", "AsyncLoopIter", "AsyncIter", "CacheAsyncLoopIter", "iterwrap"]


class Register(dict):  # pragma: no cover
    def add(self, cls):
        if cls.__name__ in self:
            warnings.warn("key %s has already existed, which will be overridden" % cls.__name__)
        self[cls.__name__] = cls
        return cls


register = Register()


@register.add
class BaseIter(object):
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
class LoopIter(BaseIter):
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


@register.add
class AsyncLoopIter(LoopIter):
    def __init__(self, src, prefetch=True, tank_size=8, timeout=None):
        self.prefetch = prefetch
        self.thread = None
        self._size = tank_size
        self.queue = queue.Queue(self._size)
        self._stop = False
        self._timeout = timeout
        super(AsyncLoopIter, self).__init__(src)

    def reset(self):
        super(AsyncLoopIter, self).reset()
        if self.thread is not None:
            self.thread.join()
        self._stop = False
        if self.prefetch:
            self.thread = threading.Thread(target=self.produce, daemon=True)
            self.thread.start()

    def __next__(self):
        if not self.prefetch:
            self.produce(False)
        if self.queue is not None:
            item = self.queue.get()
        else:
            raise StopIteration
        if isinstance(item, Exception):
            if isinstance(item, StopIteration):
                self.reset()
                raise StopIteration
            else:  # pragma: no cover
                raise item
        else:
            return item

    def produce(self, daemon=True):
        try:
            for _data in self._data:
                self._count += 1
                self.queue.put(_data)
                if daemon is False:
                    return
            raise StopIteration
        except StopIteration as e:
            if not self._stop:
                self.queue.put(e)
                self._stop = True

        except Exception as e:  # pragma: no cover
            if not self._stop:
                self._stop = True
                self.queue.put(e)


@register.add
class AsyncIter(AsyncLoopIter):
    def init(self):
        super(AsyncIter, self).reset()

    def reset(self):
        self.thread = None
        self.queue = None
        self._set_length()


@register.add
class CacheAsyncLoopIter(AsyncLoopIter):
    def __init__(self, src, cache_file=None, rerun=True, prefetch=True, tank_size=8, timeout=None):
        if cache_file is None:
            cache_dir = tempfile.mkdtemp(str(uuid.uuid4()))
            filename = str(uuid.uuid4())
            self.cache_file = path_append(cache_dir, filename)
        else:
            self.cache_file = cache_file

        self.cache_queue = None
        self.cache_thread = None

        if os.path.exists(self.cache_file) and not rerun:
            # 从已有数据中进行装载
            def src():
                return loading(self.cache_file, "json")

            self._cache_stop = True
        else:
            # 重新生成数据
            self.cache_queue = queue.Queue(tank_size)
            self.cache_thread = threading.Thread(target=self.cached, daemon=False)
            self.cache_thread.start()
            self._cache_stop = False
        super(CacheAsyncLoopIter, self).__init__(src, prefetch, tank_size, timeout)

    def init(self):
        super(CacheAsyncLoopIter, self).reset()

    def reset(self):
        self._reset = lambda: loading(self.cache_file, "json")
        if self.cache_thread is not None:
            self.cache_thread.join()
            self.cache_thread = None
            self.cache_queue = None
        super(CacheAsyncLoopIter, self).reset()

    def __next__(self):
        if not self.prefetch:
            self.produce(False)
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
    if itertype not in register:
        raise TypeError("itertype %s is unknown, the available type are %s" % (itertype, ", ".join(register)))

    def _f1(f):
        @functools.wraps(f)
        def _f2(*fargs, **fkwargs):
            return register[itertype](functools.partial(f, *fargs, **fkwargs), *args, **kwargs)

        return _f2

    return _f1
