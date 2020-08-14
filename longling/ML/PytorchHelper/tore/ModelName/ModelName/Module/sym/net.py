# coding: utf-8
# create by tongshiwei on 2019-9-1

__all__ = ["get_net", "get_bp_loss"]

# todo: define your network symbol and back propagation loss function

import torch
from torch import nn

from longling.ML.PytorchHelper.toolkit.parallel import set_device


def get_net(**kwargs):
    return NetName(**kwargs)


def get_bp_loss(ctx, **kwargs):
    return {"L2Loss": set_device(ctx, torch.nn.MSELoss(**kwargs))}


class NetName(nn.Module):
    def __init__(self, **kwargs):
        super(NetName, self).__init__()

        with self.name_scope():
            pass

    def forward(self, *args, **kwargs):
        pass
