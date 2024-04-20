# coding: utf-8
# 2020/4/19 @ tongshiwei
import itertools
import math
import random

___all__ = ["ema", "choice"]


def ema(x, mu=None, alpha=0.3):
    """
    Exponential moving average: smoothing to give progressively lower
    weights to older values.

    Code from tqdm

    Parameters
    ----------
    x  : float
        New value to include in EMA.
    mu  : float, optional
        Previous EMA value.
    alpha  : float, optional
        Smoothing factor in range [0, 1], [default: 0.3].
        Increase to give more weight to recent values.
        Ranges from 0 (yields mu) to 1 (yields x).

    Examples
    -------
    >>> ema(1)
    1
    >>> ema(1, 10)
    7.3
    """
    return x if mu is None else (alpha * x) + (1 - alpha) * mu


def smooth(array: list, smooth_weight: float) -> list:
    """
    smooth the list

    Examples
    --------
    >>> smooth([0, 1, 2, 3, 4], 0)
    [0, 1, 2, 3, 4]
    >>> "%.1f" % smooth([0, 1] * 10000, 0.999)[-1]
    '0.5'

    """
    smoothed = [array[0]]
    for scalar in array[1:]:
        smoothed.append(ema(scalar, smoothed[-1], 1 - smooth_weight))
    return smoothed


def choice(obj) -> int:
    """
    weighted choice

    Examples
    --------
    >>> random.seed(10)
    >>> choice([0.1, 0.2, 0.4, 0.3])
    2
    >>> choice({1: 0.1, 2: 0.9})
    2
    >>> choice({1: 1 / 3, 2: 0.66666})
    2
    """
    r = random.random()

    if isinstance(obj, list):
        assert obj
        if round(sum(obj), 2) != 1:
            raise ValueError("the sum of obj should be 1, now is %s" % sum(obj))
        for i, value in enumerate(itertools.accumulate(obj)):
            if r < value:
                return i
        else:  # pragma: no cover
            return obj[-1]
    elif isinstance(obj, dict):
        assert obj
        keys = obj.keys()
        values = itertools.accumulate(obj.values())
        if round(sum(obj.values()), 2) != 1:
            raise ValueError("the sum of obj should be 1, now is %s" % sum(obj.values()))
        for key, value in zip(keys, values):
            if r < value:
                return key
        else:  # pragma: no cover
            return key
    else:
        raise TypeError("obj should be either list or dict, now is %s" % type(obj))


def embedding_dim(embedding_size):
    """

    Parameters
    ----------
    embedding_size

    Returns
    -------

    Examples
    --------
    >>> embedding_dim(10e4)
    96
    """
    _constant = 8.33
    n = _constant * math.log(embedding_size)
    return int(math.ceil(n))
