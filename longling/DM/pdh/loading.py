# coding: utf-8
# create by tongshiwei on 2020-10-28

__all__ = ["parallel_read_csv"]

import pandas as pd

from longling.lib.concurrency import concurrent_pool


def parallel_read_csv(files, max_pool_size=16, params_group=None, *args, **kwargs):
    ret = []
    if params_group is None:
        with concurrent_pool("t", max(len(files), max_pool_size), ret=ret) as e:
            for file in files:
                e.submit(
                    pd.read_csv,
                    file,
                    *args,
                    **kwargs
                )
    else:
        with concurrent_pool("t", max(len(files), max_pool_size), ret=ret) as e:
            for file, params in zip(files, params_group):
                _params = {} if not params else params
                e.submit(
                    pd.read_csv,
                    file,
                    **_params
                )
    return ret
