# coding: utf-8
# create by tongshiwei on 2018/7/15
from tqdm import tqdm

import mxnet as mx
from mxnet import gluon


def get_l2_embedding_weight(F, embedding_size, batch_size=None, prefix=""):
    entries = list(range(embedding_size))
    embedding_weight = []
    batch_size = batch_size if batch_size else embedding_size
    for entity in tqdm(gluon.data.DataLoader(gluon.data.ArrayDataset(entries),
                                             batch_size=batch_size, shuffle=False),
                       'getting %s embedding' % prefix):
        embedding_weight.extend(mx.nd.L2Normalization(F(entity)).asnumpy().tolist())

    return embedding_weight
