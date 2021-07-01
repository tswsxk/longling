# coding: utf-8
# 2020/1/13 @ tongshiwei

import platform
import random

import pytest

from longling import BaseIter, MemoryIter, LoopIter, AsyncLoopIter, AsyncIter, CacheAsyncLoopIter
from longling import iterwrap, path_append


def _skip_windows():
    if platform.system() == "Windows":
        return True
    else:
        return False


def etl():
    for _ in range(100):
        yield [random.random() * 5 for _ in range(20)]


@pytest.mark.parametrize("iter_params", [(BaseIter, {}), (AsyncIter, {}), (AsyncIter, {"level": "t"})])
def test_base(iter_params):
    if _skip_windows():
        return

    Iter, params = iter_params

    @Iter.wrap
    def etl_base():
        return etl()

    data = etl_base()

    with pytest.raises(TypeError):
        len(data)

    for _ in data:
        pass

    assert len(data) == 100

    data = etl_base()
    with pytest.raises(AssertionError):
        for _ in range(3):
            tag = False
            for _ in data:
                tag = True
            else:
                assert tag

    data = Iter(etl, **params)

    for i in range(3):
        tag = False
        for _ in data:
            tag = True
        else:
            assert tag, "%s, %s" % (i, Iter)
        data.reset()


def test_loop(tmpdir):
    if _skip_windows():
        return

    test_iter = {
        LoopIter: [{}],
        AsyncLoopIter: [
            {},
            dict(level="p")
        ],
        MemoryIter: [
            {},
            dict(prefetch=True),
        ],
        CacheAsyncLoopIter: [
            dict(cache_file=path_append(tmpdir, "test.jsonl"), level="p"),
            dict(cache_file=path_append(tmpdir, "test.jsonl"), rerun=False)
        ]
    }

    for Iter, params in test_iter.items():

        if Iter != CacheAsyncLoopIter:
            @Iter.wrap
            def etl_loop():
                return etl()

            data = etl_loop()

            for _ in range(3):
                tag = False
                for _ in data:
                    tag = True
                else:
                    assert tag

        for p in params:
            @iterwrap(Iter.__name__, **p)
            def etl_loop():
                return etl()

            data = etl_loop()

            for _ in range(3):
                tag = False
                for _ in data:
                    tag = True
                else:
                    assert tag


def test_iterwrap():
    if _skip_windows():
        return

    with pytest.raises(TypeError):
        @iterwrap("a wrong case")
        def etl_loop():
            return etl()


@iterwrap(level="p")
def etl_p():
    for _ in range(100):
        yield [1, 1, 1]


def test_process_level():
    """waiting for testing"""
    if _skip_windows():
        return

    data = etl_p()
    for _ in range(3):
        tag = False
        for _ in data:
            tag = True
        else:
            assert tag
