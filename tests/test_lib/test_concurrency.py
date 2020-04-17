# coding: utf-8
# 2020/4/17 @ tongshiwei

import asyncio
import pytest
import signal
import platform
from longling import concurrent_pool


def fn(idx, bias=0):
    return idx + bias


async def async_fn(idx, bias):
    await asyncio.sleep(0.001)
    return idx + bias


def test_exception():
    with pytest.raises(TypeError):
        with concurrent_pool("c") as c:
            for i in range(4):
                c.submit(async_fn, i, bias="1")

    with pytest.raises(TypeError):
        with concurrent_pool("t") as c:
            for i in range(4):
                c.submit(fn, i, bias="1")


def test_concurrency():
    ret = []
    with concurrent_pool("c", ret=ret) as c:
        for i in range(4):
            c.submit(async_fn, i, bias=1)

    assert list(sorted(ret)) == [1, 2, 3, 4]

    ret = []
    with concurrent_pool("t", ret=ret) as c:
        for i in range(4):
            c.submit(fn, i, bias=1)

    assert list(sorted(ret)) == [1, 2, 3, 4]

    if "Windows" in platform.architecture()[1]:
        def shutdown(frame, signum):
            # your app's shutdown or whatever
            pass

        signal.signal(signal.SIGBREAK, shutdown)

        try:
            from pytest_cov.embed import cleanup_on_signal
        except ImportError:
            pass
        else:
            cleanup_on_signal(signal.SIGBREAK)
        return
    else:
        try:
            from pytest_cov.embed import cleanup_on_sigterm
        except ImportError:
            pass
        else:
            cleanup_on_sigterm()

    ret = []
    with concurrent_pool("p", ret=ret) as c:
        for i in range(4):
            c.submit(fn, i, bias=1)

    assert list(sorted(ret)) == [1, 2, 3, 4]
