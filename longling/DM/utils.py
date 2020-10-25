# coding: utf-8
# create by tongshiwei on 2020-10-17

import scipy.stats
import pandas as pd

from longling import as_list


def describe(array, op: (str, list) = None):
    """

    Parameters
    ----------
    array
    op

    Returns
    -------

    Examples
    ---------
    >>> a = range(10)
    >>> describe(a)
    {'mean': 4.5, 'min': 0, 'max': 9, 'var': 9.166666666666666, 'skew': 0.0, 'kurtosis': -1.2242424242424244}
    """
    result = {}

    op = as_list(op) if op is not None else [
        "mean", "max", "min", "var"
    ]

    sci_dis = scipy.stats.describe(array)

    result.update({
        "mean": sci_dis.mean,
        "min": sci_dis.minmax[0],
        "max": sci_dis.minmax[1],
        "var": sci_dis.variance,
        "skew": sci_dis.skewness,
        "kurtosis": sci_dis.kurtosis,
    })

    res = pd.Series(array).describe()

    return result
