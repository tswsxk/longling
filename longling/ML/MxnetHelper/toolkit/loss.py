# coding: utf-8
# create by tongshiwei on 2020-11-19

__all__ = ["as_tmt_mx_loss", "loss_dict2tmt_mx_loss"]

import mxnet.ndarray as nd

from longling.ML.toolkit import as_tmt_loss, loss_dict2tmt_loss


def _loss2value(loss):
    return nd.mean(loss).asscalar()


# def tmt_mx_loss(loss2value=_loss2value):
#     return tmt_loss(loss2value)


def as_tmt_mx_loss(loss_obj, loss2value=_loss2value):
    return as_tmt_loss(loss_obj, loss2value)


def loss_dict2tmt_mx_loss(loss_dict, loss2value=_loss2value, exclude=None, include=None):
    return loss_dict2tmt_loss(loss_dict, loss2value, exclude, include, as_loss=as_tmt_mx_loss)
