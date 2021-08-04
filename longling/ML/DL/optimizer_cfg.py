# coding: utf-8
# create by tongshiwei on 2020-11-17

from copy import deepcopy

__all__ = ["get_optimizer_cfg", "OPTIMIZERS"]

# By default, we suppose the batch_size is 16
# if the batch_size is 128, learning_rate should be 10 times default value
# e.g., for Adam, learning rate should be changed from 1e-3 to 1e-2.

OPTIMIZERS = {
    "base": (
        "Adam",
        {
            "learning_rate": 1e-3,
            "wd": 1e-4,
            "clip_gradient": 1,
        }
    ),
    "recurrent": (
        "RMSProp",
        {
            "learning_rate": 1e-4,
            "wd": 1e-4,
            "clip_gradient": 1,
        }
    ),
    "stable": (
        "sgd",
        {
            "learning_rate": 1e-2,
            "wd": 1e-4,
            "clip_gradient": 1,
            "momentum": 0.9,
        }
    ),
    "word_embedding": (
        "adagrad",
        {
            "learning_rate": 1e-2,
            "wd": 1e-4,
            "clip_gradient": 1,
        }
    )
}


def get_optimizer_cfg(name) -> tuple:
    """

    Parameters
    ----------
    name

    Returns
    -------
    optimizer_name: str
    optimizer_params: dict

    Examples
    --------
    >>> get_optimizer_cfg('base')
    ('Adam', {'learning_rate': 0.001, 'wd': 0.0001, 'clip_gradient': 1})
    """
    try:
        optimizer, optimizer_params = OPTIMIZERS[name]
    except KeyError:
        raise KeyError("Unknown optimizer name: %s, the name should be in: %s" % (name, ", ".join(OPTIMIZERS)))
    optimizer_params = deepcopy(optimizer_params)

    return optimizer, optimizer_params
