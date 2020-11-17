# coding: utf-8
# create by tongshiwei on 2020-11-17

from copy import deepcopy

__all__ = ["get_optimizer_cfg"]

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


def get_optimizer_cfg(name):
    try:
        optimizer, optimizer_params = optimizers[name]
    except KeyError:
        raise KeyError("the name should be in: %s" % ", ".join(optimizers))
    optimizer_params = deepcopy(optimizer_params)

    return optimizer, optimizer_params
