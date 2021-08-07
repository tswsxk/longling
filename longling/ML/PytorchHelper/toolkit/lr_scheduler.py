# coding: utf-8
# 2021/3/15 @ tongshiwei
import logging
from torch.optim.lr_scheduler import *


def get_lr_scheduler(scheduler: (str, ...) = "cosine", logger=logging, update_params=None, adaptive=True, *, optimizer,
                     **kwargs):
    """

    Parameters
    ----------
    scheduler
    logger
    update_params
    adaptive
    optimizer
    kwargs

    Returns
    -------

    Examples
    --------
    >>> import torch
    >>> scheduler = get_lr_scheduler(
    ...     "OneCycleLR",
    ...     max_lr=1,
    ...     total_steps=100,
    ...     optimizer=torch.optim.Adam(torch.nn.Linear(10, 30).parameters(), 0.1)
    ... )
    >>> scheduler.get_last_lr()[0]  # doctest: +ELLIPSIS
    0.040...
    >>> for _ in range(50):
    ...     scheduler.step()
    >>> scheduler.get_last_lr()[0]  # doctest: +ELLIPSIS
    0.793...
    >>> for _ in range(50):
    ...     scheduler.step()
    >>> scheduler.get_last_lr()[0]  # doctest: +ELLIPSIS
    0.0005...
    """
    if isinstance(scheduler, str):
        scheduler = eval(scheduler)(optimizer=optimizer, **kwargs)

    logger.info(scheduler)

    return scheduler


def plot_schedule(scheduler, iterations=1500):  # pragma: no cover
    import matplotlib.pyplot as plt
    lrs = []
    for _ in range(iterations):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    plt.scatter(list(range(iterations)), lrs)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.show()


if __name__ == '__main__':
    import torch

    logging.getLogger().setLevel(logging.INFO)
    _scheduler = get_lr_scheduler(
        "OneCycleLR", max_lr=1, total_steps=1500, optimizer=torch.optim.Adam(torch.nn.Linear(10, 30).parameters(), 0.1)
    )
    plot_schedule(_scheduler)
