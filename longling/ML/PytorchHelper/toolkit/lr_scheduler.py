# coding: utf-8
# 2021/3/15 @ tongshiwei
import logging
from torch.optim.lr_scheduler import *


def get_lr_scheduler(scheduler: (str, ...) = "cosine", logger=logging, update_params=None, adaptive=True, *, optimizer,
                     **kwargs):
    if isinstance(scheduler, str):
        scheduler = eval(scheduler)(optimizer=optimizer, **kwargs)

    logger.info(scheduler)

    return scheduler


def plot_schedule(scheduler, iterations=1500):
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
        "OneCycleLR", max_lr=1, total_steps=10000, optimizer=torch.optim.Adam(torch.nn.Linear(10, 30).parameters(), 0.1)
    )
    plot_schedule(_scheduler)
