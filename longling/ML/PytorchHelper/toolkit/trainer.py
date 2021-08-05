# coding: utf-8
# create by tongshiwei on 2019-8-30
import re

import torch
# from .lr_scheduler import get_lr_scheduler
from longling.ML.PytorchHelper.toolkit.lr_scheduler import get_lr_scheduler


def collect_params(net, select=None):
    """

    Parameters
    ----------
    net
    select

    Returns
    -------
    >>> class EmbedMLP(torch.nn.Module):
    ...     def __init__(self):
    ...         super(EmbedMLP, self).__init__()
    ...         self.embedding = torch.nn.Embedding(5, 3)
    ...         self.linear = torch.nn.Linear(3, 5)
    >>> net = EmbedMLP()
    >>> len(list(collect_params(net, None)))
    3
    >>> from longling.ML.DL import ALL_PARAMS, BLOCK_EMBEDDING
    >>> len(list(collect_params(net, ALL_PARAMS)))
    3
    >>> len(list(collect_params(net, BLOCK_EMBEDDING)))
    2
    >>> len(list(collect_params(net.linear)))
    2
    >>> len(list(collect_params(net.embedding)))
    1
    """
    if select is None:
        return net.parameters()
    pattern = re.compile(select)
    ret = [value for name, value in net.named_parameters() if pattern.match(name)]
    return ret


def get_trainer(net, optimizer, optimizer_params=None, lr_params=None, select=None) -> ...:
    """

    Parameters
    ----------
    net
    optimizer
    optimizer_params
    lr_params
    select

    Returns
    -------

    Examples
    >>> class EmbedMLP(torch.nn.Module):
    ...     def __init__(self):
    ...         super(EmbedMLP, self).__init__()
    ...         self.embedding = torch.nn.Embedding(5, 3)
    ...         self.linear = torch.nn.Linear(3, 5)
    >>> net = EmbedMLP()
    >>> optimizer = get_trainer(net, "sgd", optimizer_params={"lr": 0.01})
    >>> optimizer
    SGD (
    Parameter Group 0
        dampening: 0
        lr: 0.01
        momentum: 0
        nesterov: False
        weight_decay: 0
    )
    >>> len(optimizer.param_groups)
    1
    >>> sum([len(group["params"]) for group in optimizer.param_groups])
    3
    >>> from longling.ML.DL import BLOCK_EMBEDDING
    >>> optimizer = get_trainer(net, "sgd", optimizer_params={"lr": 0.01}, select=BLOCK_EMBEDDING)
    >>> len(optimizer.param_groups)
    1
    >>> sum([len(group["params"]) for group in optimizer.param_groups])
    2
    >>> optimizer, lr_scheduler = get_trainer(
    ...     net, "sgd",
    ...     optimizer_params={"lr": 0.01},
    ...     lr_params={"scheduler": "OneCycleLR", "max_lr": 1, "total_steps": 100}
    ... )
    >>> optimizer
    SGD (
    Parameter Group 0
        base_momentum: 0.85
        dampening: 0
        initial_lr: 0.04
        lr: 0.040000000000000036
        max_lr: 1
        max_momentum: 0.95
        min_lr: 4e-06
        momentum: 0.95
        nesterov: False
        weight_decay: 0
    )
    >>> lr_scheduler  # doctest: +ELLIPSIS
    <torch.optim.lr_scheduler.OneCycleLR...
    >>> get_trainer(net, "adam", optimizer_params={"lr": 0.01})
    Adam (
    Parameter Group 0
        amsgrad: False
        betas: (0.9, 0.999)
        eps: 1e-08
        lr: 0.01
        weight_decay: 0
    )
    >>> get_trainer(net, "adagrad", optimizer_params={"lr": 0.01})
    Adagrad (
    Parameter Group 0
        eps: 1e-10
        initial_accumulator_value: 0
        lr: 0.01
        lr_decay: 0
        weight_decay: 0
    )
    >>> get_trainer(net, "rmsprop", optimizer_params={"lr": 0.01})
    RMSprop (
    Parameter Group 0
        alpha: 0.99
        centered: False
        eps: 1e-08
        lr: 0.01
        momentum: 0
        weight_decay: 0
    )
    >>> get_trainer(net, "Rprop", optimizer_params={"lr": 0.01})
    Rprop (
    Parameter Group 0
        etas: (0.5, 1.2)
        lr: 0.01
        step_sizes: (1e-06, 50)
    )
    >>> import torch
    >>> get_trainer(net, torch.optim.SGD, optimizer_params={"lr": 0.01})
    SGD (
    Parameter Group 0
        dampening: 0
        lr: 0.01
        momentum: 0
        nesterov: False
        weight_decay: 0
    )
    """
    assert isinstance(optimizer, str) or issubclass(optimizer, torch.optim.Optimizer)
    parameters = collect_params(net, select)
    assert optimizer_params is None or isinstance(optimizer_params, dict)
    optimizer_params = {} if optimizer_params is None else optimizer_params

    if isinstance(optimizer, str):
        if optimizer == "sgd":
            optimizer_class = torch.optim.SGD
        elif optimizer == "adam":
            optimizer_class = torch.optim.Adam
        elif optimizer == "adagrad":
            optimizer_class = torch.optim.Adagrad
        elif optimizer == "rmsprop":
            optimizer_class = torch.optim.RMSprop
        else:
            optimizer_class = eval("torch.optim.%s" % optimizer)
    else:
        optimizer_class = optimizer

    optimizer = optimizer_class(parameters, **optimizer_params)
    if lr_params:
        lr_scheduler = get_lr_scheduler(optimizer=optimizer, **lr_params)
        return optimizer, lr_scheduler
    else:
        return optimizer
