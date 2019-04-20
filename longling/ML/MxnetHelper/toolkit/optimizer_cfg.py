# coding: utf-8
# create by tongshiwei on 2019/4/12
from copy import deepcopy

import mxnet as mx

__all__ = ["get_optimizer_cfg", "get_lr_scheduler", "get_update_steps"]

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
    ),
    "word_embedding": (
        "adagrad",
        {
            "learning_rate": 10e-2,
            "wd": 10e-4,
            "clip_gradient": 1,
        }
    )
}


def get_update_steps(update_epoch, batches_per_epoch):
    return update_epoch * batches_per_epoch


def get_lr_scheduler(learning_rate, step=None, max_update_steps=None,
                     discount=0.01):
    factor = pow(discount, step / max_update_steps)
    return mx.lr_scheduler.FactorScheduler(
        step, factor,
        stop_factor_lr=learning_rate * discount
    )


def get_optimizer_cfg(name):
    try:
        optimizer, optimizer_params = optimizers[name]
    except KeyError:
        raise KeyError("the name should be in: %s" % ", ".join(optimizers))
    optimizer_params = deepcopy(optimizer_params)

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
