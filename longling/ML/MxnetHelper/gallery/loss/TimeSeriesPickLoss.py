# coding:utf-8
# created by tongshiwei on 2018/7/11
"""
This file contains some loss function
"""
import mxnet as mx
from mxnet import gluon

__all__ = ["TimeSeriesPickLoss"]


class TimeSeriesPickLoss(gluon.HybridBlock):
    """
    Loss from Deep Knowledge Tracing

    Notes
    -----
    The loss has been average, so when call the step method of trainer,
    batch_size should be 1
    """

    def hybrid_forward(self, F, pred, pick_index, label, label_mask, *args,
                       **kwargs):
        pred = F.slice(pred, (None, None), (None, -1))
        pred = F.pick(pred, pick_index)
        loss = F.LogisticRegressionOutput(pred, label)
        loss = F.sum(mx.nd.squeeze(
            F.SequenceMask(F.expand_dims(loss, 0), sequence_length=label_mask,
                           use_sequence_length=True)), axis=-1)
        loss = F.mean(loss)
        return loss
