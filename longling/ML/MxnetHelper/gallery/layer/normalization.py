# coding: utf-8
# create by tongshiwei on 2019/4/7

import mxnet as mx
from mxnet import gluon

__all__ = ["get_l2_embedding_weight"]


def get_l2_embedding_weight(F, embedding_size, batch_size=None):
    entries = list(range(embedding_size))
    embedding_weight = []
    batch_size = batch_size if batch_size else embedding_size
    for entity in gluon.data.DataLoader(gluon.data.ArrayDataset(entries),
                                        batch_size=batch_size,
                                        shuffle=False):
        embedding_weight.extend(
            mx.nd.L2Normalization(F(entity)).asnumpy().tolist())

    return embedding_weight
