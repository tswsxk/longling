# coding: utf-8
# create by tongshiwei on 2019-8-30
import re

import torch
from .lr_scheduler import get_lr_scheduler


def collect_params(_net, select):
    if select is None:
        return _net.parameters()
    pattern = re.compile(select)
    ret = [value for name, value in _net.named_parameters() if pattern.match(name)]
    return ret


def get_trainer(_net, optimizer, optimizer_params=None, lr_params=None, select=None):
    assert isinstance(optimizer, str) or issubclass(optimizer, torch.optim.Optimizer)
    parameters = collect_params(_net, select)
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
