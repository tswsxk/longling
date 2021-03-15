# coding: utf-8
# 2021/3/15 @ tongshiwei
import re
from collections import OrderedDict
import torch


def save_params(filename, net: torch.nn.Module, select=None):
    params = net.state_dict()

    if select is None:
        if select is not None:
            pattern = re.compile(select)
            params = OrderedDict({name: value for name, value in params.items() if
                                  pattern.match(name)})

    torch.save(params, filename)
