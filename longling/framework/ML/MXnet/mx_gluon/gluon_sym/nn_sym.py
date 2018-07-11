# coding:utf-8
# created by tongshiwei on 2018/7/11
from mxnet import gluon


class PairwiseLoss(gluon.loss.Loss):
    def hybrid_forward(self, F, pos, neg, *args, **kwargs):
        loss = F.add_n(F.negative(pos), neg)
        sym = F.relu(loss)
        loss = F.MakeLoss(sym)
        return loss
