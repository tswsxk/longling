# coding: utf-8
# 2021/3/15 @ tongshiwei

from copy import deepcopy
from longling.ML.DL.optimizer_cfg import OPTIMIZERS as _OPTIMIZERS

OPTIMIZERS = deepcopy(_OPTIMIZERS)

for optimizer in OPTIMIZERS.values():
    params = optimizer[1]
    params["lr"] = params.pop("learning_rate")
    params["weight_decay"] = params.pop("wd")
    params.pop("clip_gradient")


def get_optimizer_cfg(name):
    """
    Examples
    -------
    >>> get_optimizer_cfg('base')
    ('Adam', {'lr': 0.001, 'weight_decay': 0.0001})
    """
    try:
        optimizer, optimizer_params = OPTIMIZERS[name]
    except KeyError:
        raise KeyError("the name should be in: %s" % ", ".join(OPTIMIZERS))
    optimizer_params = deepcopy(optimizer_params)

    return optimizer, optimizer_params
