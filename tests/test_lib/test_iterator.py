# coding: utf-8
# 2020/1/13 @ tongshiwei

import pytest
import random
from longling import BaseIter, LoopIter, AsyncLoopIter, AsyncIter, CacheAsyncLoopIter
from longling import iterwrap, path_append


def etl():
    for _ in range(100):
        yield [random.random() * 5 for _ in range(20)]


def test_base():
    for Iter in [BaseIter, AsyncIter]:
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


def test_loop(tmpdir):
    test_iter = {
        LoopIter: [],
        AsyncLoopIter: [dict(prefetch=False)],
        CacheAsyncLoopIter: [
            dict(prefetch=False),
            dict(cache_file=path_append(tmpdir, "test.json")),
            dict(cache_file=path_append(tmpdir, "test.json"), rerun=False)
        ]
    }

    for Iter, params in test_iter.items():
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
    with pytest.raises(TypeError):
        @iterwrap("a wrong case")
        def etl_loop():
            return etl()
