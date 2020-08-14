# coding: utf-8
# 2020/4/19 @ tongshiwei


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
