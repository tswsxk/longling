# coding: utf-8
# 2020/4/19 @ tongshiwei
import random
import itertools

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
    Examples
    --------
    >>> random.seed(10)
    >>> choice([0.1, 0.2, 0.4, 0.3])
    2
    >>> choice({1: 0.1, 2: 0.9})
    2
    """
    r = random.random()
    if isinstance(obj, list):
        for i, value in enumerate(itertools.accumulate(obj)):
            if r < value:
                return i
        else:
            raise ValueError("the sum of obj should be 1, now is %s" % sum(obj))
    elif isinstance(obj, dict):
        keys = obj.keys()
        values = itertools.accumulate(obj.values())
        for key, value in zip(keys, values):
            if r < value:
                return key
        else:
            raise ValueError("the sum of obj should be 1, now is %s" % sum(values))
    else:
        raise TypeError("obj should be either list or dict, now is %s" % type(obj))