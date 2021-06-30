# coding: utf-8
# 2021/3/13 @ tongshiwei


__all__ = ["as_tmt_torch_loss", "loss_dict2tmt_torch_loss"]

import torch
from longling.ML.toolkit import as_tmt_loss, loss_dict2tmt_loss


def _loss2value(loss: torch.Tensor):
    return loss.mean().item()


# def tmt_torch_loss(loss2value=_loss2value):
#     return tmt_loss(loss2value)


def as_tmt_torch_loss(loss_obj, loss2value=_loss2value):
    return as_tmt_loss(loss_obj, loss2value)


def loss_dict2tmt_torch_loss(loss_dict, loss2value=_loss2value, exclude=None, include=None):
    return loss_dict2tmt_loss(loss_dict, loss2value, exclude, include, as_loss=as_tmt_torch_loss)
