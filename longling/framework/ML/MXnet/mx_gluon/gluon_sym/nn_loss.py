# coding:utf-8
# created by tongshiwei on 2018/7/11
"""
This file contains some loss function
"""
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.loss import *


class PairwiseLoss(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=-1, margin=0, **kwargs):
        super(PairwiseLoss, self).__init__(weight, batch_axis)
        self.margin = margin

    def hybrid_forward(self, F, pos, neg, *args, **kwargs):
        loss = F.add_n(F.negative(pos), neg)
        loss = F.add(loss, self.margin)
        sym = F.relu(loss)
        loss = F.MakeLoss(sym)
        return loss


# class TimeSeriesLoss(gluon.HybridBlock):
#     def hybrid_forward(self, F, pred, label, mask, *args, **kwargs):
#         F.stack()


class TimeSeriesPickLoss(gluon.HybridBlock):
    """
    Loss from Deep Knowledge Tracing

    Notes
    -----
    The loss has been average, so when call the step method of trainer, batch_size should be 1
    """

    def hybrid_forward(self, F, pred, pick_index, label, label_mask, *args, **kwargs):
        pred = F.slice(pred, (None, None), (None, -1))
        pred = F.pick(pred, pick_index)
        loss = F.LogisticRegressionOutput(pred, label)
        loss = F.sum(mx.nd.squeeze(
            F.SequenceMask(F.expand_dims(loss, 0), sequence_length=label_mask,
                           use_sequence_length=True)), axis=-1)
        loss = F.mean(loss)
        return loss
