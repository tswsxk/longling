# coding:utf-8
# created by tongshiwei on 2018/7/11
from mxnet import gluon


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