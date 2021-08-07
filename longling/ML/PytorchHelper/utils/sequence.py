# coding: utf-8
# create by tongshiwei on 2019-8-30

__all__ = ["length2mask", "get_sequence_mask", "sequence_mask"]

import torch
from torch import Tensor

from .utils import tensor2list


def length2mask(length, max_len, valid_mask_val, invalid_mask_val):
    """
    >>> data_len = [1, 2, 3]
    >>> length2mask(data_len, 5, 1, 0)
    tensor([[1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0]])
    """
    mask = []

    if isinstance(valid_mask_val, Tensor):
        valid_mask_val = tensor2list(valid_mask_val)
    if isinstance(invalid_mask_val, Tensor):
        invalid_mask_val = tensor2list(invalid_mask_val)
    if isinstance(length, Tensor):
        length = tensor2list(length)

    for _len in length:
        mask.append([valid_mask_val] * _len + [invalid_mask_val] * (max_len - _len))

    return torch.tensor(mask)


def get_sequence_mask(shape, sequence_length, axis=1):
    """
    Parameters
    ----------
    shape:
    sequence_length
    axis: int
        the length axis,
        if there is a batch axis, should be placed before length, like (batch, length, ...)
        instead of (length, batch, ...)

    Examples
    --------
    >>> data_shape = (2, 3, 4)
    >>> mask = get_sequence_mask(data_shape, [1, 3])
    >>> mask.shape
    torch.Size([2, 3, 4])
    >>> mask
    tensor([[[1., 1., 1., 1.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.]],
    <BLANKLINE>
            [[1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [1., 1., 1., 1.]]])
    """
    assert axis <= len(shape)
    mask_shape = shape[axis + 1:]

    valid_mask_val = torch.ones(mask_shape)
    invalid_mask_val = torch.zeros(mask_shape)

    max_len = shape[axis]

    return length2mask(sequence_length, max_len, valid_mask_val, invalid_mask_val)


def sequence_mask(tensor: Tensor, sequence_length, axis=1):
    """
    Parameters
    ----------
    tensor: Tensor
    sequence_length
    axis: int
        the length axis,
        if there is a batch axis, should be placed before length, like (batch, length, ...)
        instead of (length, batch, ...)

    Returns
    -------
    masked_tensor: Tensor

    Examples
    --------
    >>> seq = torch.tensor([[1, 1, 2], [3, 1, 0]])
    >>> sequence_mask(seq, [2, 1])
    tensor([[1., 1., 0.],
            [3., 0., 0.]])
    >>> sequence_mask(seq, torch.tensor([2, 1]))
    tensor([[1., 1., 0.],
            [3., 0., 0.]])
    """
    mask = get_sequence_mask(tensor.shape, sequence_length, axis).to(tensor.device)
    return tensor * mask
