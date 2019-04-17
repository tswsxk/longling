# coding: utf-8
# create by tongshiwei on 2019/4/7
import math

import mxnet as mx
from mxnet import gluon

__all__ = ["TransE"]


class TransE(gluon.HybridBlock):
    def __init__(self,
                 entities_size, relations_size, dim=50,
                 d='L1', **kwargs):
        super(TransE, self).__init__(**kwargs)
        self.d = d

        with self.name_scope():
            self.entity_embedding = gluon.nn.Embedding(
                entities_size, dim,
                weight_initializer=mx.init.Uniform(6 / math.sqrt(dim))
            )

            self.relation_embedding = gluon.nn.Embedding(
                relations_size, dim,
                weight_initializer=mx.init.Uniform(6 / math.sqrt(dim))
            )

            self.batch_norm = gluon.nn.BatchNorm()

    def hybrid_forward(self, F, sub, rel, obj, **kwargs):
        sub = self.entity_embedding(sub)
        rel = self.relation_embedding(rel)
        obj = self.entity_embedding(obj)

        sub = F.L2Normalization(sub)
        rel = F.L2Normalization(rel)
        obj = F.L2Normalization(obj)

        sub = self.batch_norm(sub)
        rel = self.batch_norm(rel)
        obj = self.batch_norm(obj)
        distance = F.add_n(sub, rel, F.negative(obj))
        if self.d == 'L2':
            return F.norm(distance, axis=1)
        elif self.d == 'L1':
            return F.sum(F.abs(distance), axis=1)
