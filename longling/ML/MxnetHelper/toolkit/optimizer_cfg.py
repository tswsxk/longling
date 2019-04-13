# coding: utf-8
# create by tongshiwei on 2019/4/12

__all__ = ["get_optimizer_cfg"]

import warnings
from copy import deepcopy

import mxnet as mx

optimizers = {
    "base": (
        "Adam",
        {
            "learning_rate": 10e-3,
            "wd": 10e-4,
            "clip_gradient": 1,
        }
    ),
    "recurrent": (
        "RMSProp",
        {
            "learning_rate": 10e-4,
            "wd": 10e-4,
            "clip_gradient": 1,
        }
    ),
    "stable": (
        "sgd",
        {
            "learning_rate": 10e-2,
            "wd": 10e-4,
            "clip_gradient": 1,
            "momentum": 0.9,
        }
    )
}


def get_optimizer_cfg(name, lr_scheduler=None, lr_step=None, lr_total_step=None):
    try:
        optimizer, optimizer_params = optimizers[name]
    except KeyError:
        raise KeyError("the name should be in: %s" % ", ".join(optimizers))
    optimizer_params = deepcopy(optimizer_params)
    discount = 0.01
    if lr_scheduler is not None:
        optimizer_params["lr_scheduler"] = lr_scheduler
        if all([lr_step, lr_total_step]):
            warnings.warn(
                "step and factor are invalid due to lr_scheduler is not None"
            )
    elif all([lr_step, lr_total_step]):
        factor = pow(discount, lr_step / lr_total_step)
        lr_scheduler = mx.lr_scheduler.FactorScheduler(
            lr_step, factor,
            stop_factor_lr=optimizer_params["learning_rate"] * discount
        )
        optimizer_params["lr_scheduler"] = lr_scheduler

    return optimizer, optimizer_params


def plot_schedule(schedule_fn, iterations=1500):
    import matplotlib.pyplot as plt
    # Iteration count starting at 1
    iterations = [i + 1 for i in range(iterations)]
    lrs = [schedule_fn(i) for i in iterations]
    plt.scatter(iterations, lrs)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.show()


if __name__ == '__main__':
    _, params = get_optimizer_cfg("base", lr_step=10, lr_total_step=1000)
    plot_schedule(params["lr_scheduler"])
