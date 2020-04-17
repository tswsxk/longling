# coding: utf-8
# 2020/4/17 @ tongshiwei

import asyncio
from concurrent.futures import ThreadPoolExecutor as StdTPE
from multiprocessing.pool import Pool

PROCESS_LEVEL = 0
THREAD_LEVEL = 1
COROUTINE = 2

_LEVEL = dict(
    t=THREAD_LEVEL,
    thread=THREAD_LEVEL,
    threading=THREAD_LEVEL,
    p=PROCESS_LEVEL,
    process=PROCESS_LEVEL,
    processing=PROCESS_LEVEL,
    c=COROUTINE,
    coro=COROUTINE,
    coroutine=COROUTINE,
)


class ThreadPoolExecutor(StdTPE):
    def __init__(self, max_workers=None, thread_name_prefix='', ret: list = None):
        """Initializes a new ThreadPoolExecutor instance.

        Args:
            max_workers: The maximum number of threads that can be used to
                execute the given calls.
            thread_name_prefix: An optional name prefix to give our threads.
        """
        super(ThreadPoolExecutor, self).__init__(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
        self._result = []
        self.ret = ret

    def submit(self, fn, *args, **kwargs):
        _result = super(ThreadPoolExecutor, self).submit(fn, *args, **kwargs)
        self._result.append(_result)
        return _result

    def __exit__(self, exc_type, exc_val, exc_tb):
        for r in self._result:
            if r.exception() is not None:
                raise r.exception()
            elif self.ret is not None:
                self.ret.append(r.result())

        super(ThreadPoolExecutor, self).__exit__(exc_type, exc_val, exc_tb)


class ProcessPool(Pool):
    def __init__(self, processes=None, *args, ret: list = None, **kwargs):
        super(ProcessPool, self).__init__(processes, *args, **kwargs)
        self._result = []
        self.ret = ret

    def submit(self, func, *args, **kwargs):
        _result = self.apply_async(func, args, kwargs)
        self._result.append(_result)
        return _result

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ret is not None:
            for r in self._result:
                self.ret.append(r.get())
        super(ProcessPool, self).__exit__(exc_type, exc_val, exc_tb)


class CoroutinePool(object):
    def __init__(self, *args, ret: list = None, **kwargs):
        self.pools = []
        self.ret = ret
        self._e = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        loop = asyncio.get_event_loop()
        self._e = asyncio.gather(*[func(*arg, **kwd) for func, arg, kwd in self.pools])
        loop.run_until_complete(self._e)
        loop.close()
        if self.ret is not None:
            for r in self._e.result():
                self.ret.append(r)

    def submit(self, func, *args, **kwargs):
        return self.pools.append([func, args, kwargs])


LEVEL_CLASS = {
    PROCESS_LEVEL: ProcessPool,
    THREAD_LEVEL: ThreadPoolExecutor,
    COROUTINE: CoroutinePool,
}


def concurrent_pool(level, pool_size=None, ret=None):
    """
    Simple api for start completely independent concurrent programs:
    * thread
    * process
    * coroutine

    Examples
    --------
    .. code-block :: python

        def pseudo(idx):
            return idx

    .. code-block :: python

        ret = []
        with concurrent_pool("p", ret=ret) as e:  # or concurrent_pool("t", ret=ret)
             for i in range(4):
                e.submit(pseudo, i)
        print(ret)

    ``[0, 1, 2, 3]``


    """
    _level = _LEVEL[level]

    return LEVEL_CLASS[_level](pool_size, ret=ret)
