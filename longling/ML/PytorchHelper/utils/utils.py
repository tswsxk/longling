# coding: utf-8
# 2021/8/5 @ tongshiwei


__all__ = ["pick", "tensor2list"]

import torch
from torch import Tensor


def pick(tensor: Tensor, index: Tensor, axis=-1):
    """

    Parameters
    ----------
    tensor
    index
    axis

    Returns
    -------
    >>> data = torch.tensor([[111, 112], [121, 122], [131, 132]])
    >>> index = torch.tensor([0, 1, 0])
    >>> pick(data, index)
    tensor([111, 122, 131])
    >>> data = torch.tensor([[[111, 112], [121, 122], [131, 132]], [[211, 212], [221, 222], [231, 232]]])
    >>> index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    >>> pick(data, index)
    tensor([[111, 122, 131],
            [212, 221, 231]])
    """
    return torch.gather(tensor, axis, index.unsqueeze(axis)).squeeze(axis)


def tensor2list(tensor: Tensor):
    """
    Examples
    -------
    >>> tensor2list(torch.tensor([1, 2, 3]))
    [1, 2, 3]
    """
    return tensor.cpu().tolist()
