# coding: utf-8
# create by tongshiwei on 2019-8-30

__all__ = ["pick", "tensor2list"]

import torch
from torch import Tensor


def pick(tensor, index, axis=-1):
    return torch.gather(tensor, axis, index.unsqueeze(axis)).squeeze(axis)


def tensor2list(tensor: Tensor):
    return tensor.cpu().tolist()
