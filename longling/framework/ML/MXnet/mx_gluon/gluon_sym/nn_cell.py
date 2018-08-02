# coding: utf-8
# created by tongshiwei on 18-2-6
import math

import mxnet as mx
from mxnet import gluon


class TextCNN(gluon.HybridBlock):
    def __init__(self, sentence_size, vec_size, channel_size=None, num_output=2, filter_list=[1, 2, 3, 4],
                 num_filter=60,
                 dropout=0.0, batch_norms=0, highway=True, activation="relu", pool_type='max',
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.sentence_size = sentence_size
        self.vec_size = vec_size
        self.channel_size = channel_size
        self.num_output = num_output
        self.filter_list = filter_list
        self.num_filter = num_filter
        self.dropout_p = dropout
        self.batch_norms = batch_norms
        self.highway = highway
        self.activation = activation

        self.conv = [0] * len(self.filter_list)
        self.pool = [0] * len(self.filter_list)
        self.bn = [0] * len(self.filter_list)

        self.dropout = None

        pool2d = gluon.nn.MaxPool2D if pool_type == "max" else gluon.nn.AvgPool2D
        pool3d = gluon.nn.MaxPool3D if pool_type == "max" else gluon.nn.AvgPool3D

        with self.name_scope():
            for i, filter_size in enumerate(self.filter_list):
                conv = gluon.nn.Conv2D(self.num_filter, kernel_size=(filter_size, self.vec_size),
                                       activation=self.activation) if not self.channel_size else gluon.nn.Conv3D(
                    self.num_filter, kernel_size=(filter_size, self.vec_size, self.channel_size), activation=activation)
                setattr(self, "conv%s" % i, conv)

                pool = pool2d(pool_size=(self.sentence_size - filter_size + 1, 1),
                                          strides=(1, 1)) if not self.channel_size else pool3d(
                    pool_size=(self.sentence_size - filter_size + 1, 1, 1), strides=(1, 1, 1))
                setattr(self, "pool%s" % i, pool)
                if self.batch_norms > 0:
                    setattr(self, "bn%s" % i, gluon.nn.BatchNorm())
            if self.highway:
                self.high_fc = gluon.nn.Dense(len(filter_list) * self.num_filter, activation="relu")
                self.high_trans_fc = gluon.nn.Dense(len(filter_list) * self.num_filter, activation="sigmoid")

            if self.dropout_p > 0:
                self.dropout = gluon.nn.Dropout(self.dropout_p)

            self.fc = gluon.nn.Dense(num_output)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = F.expand_dims(x, axis=1)

        pooled_outputs = []
        for i, _ in enumerate(self.filter_list):
            convi = getattr(self, "conv%s" % i)(x)
            pooli = getattr(self, "pool%s" % i)(convi)
            if self.batch_norms > 0:
                pooli = getattr(self, "bn%s" % i)(pooli)
            pooled_outputs.append(pooli)

        total_filters = self.num_filter * len(self.filter_list)
        concat = F.Concat(dim=1, *pooled_outputs)
        h_pool = F.Reshape(data=concat, shape=(-1, total_filters))
        if self.highway:
            h_pool = highway_cell(h_pool, self.high_fc, self.high_trans_fc)

        if self.dropout_p > 0.0:
            h_drop = self.dropout(h_pool)
        else:
            h_drop = h_pool

        fc = self.fc(h_drop)

        return fc


def highway_cell(data, high_fc, high_trans_fc):
    _data = data

    high_relu = high_fc(_data)
    high_trans_sigmoid = high_trans_fc(_data)

    return high_relu * high_trans_sigmoid + _data * (1 - high_trans_sigmoid)


class TransE(gluon.HybridBlock):
    def __init__(self,
                 entities_size, relations_size, dim=50,
                 d='L1', **kwargs):
        super(TransE, self).__init__(**kwargs)
        self.d = d

        with self.name_scope():
            self.entity_embedding = gluon.nn.Embedding(entities_size, dim,
                                                       weight_initializer=mx.init.Uniform(6 / math.sqrt(dim)))

            self.relation_embedding = gluon.nn.Embedding(relations_size, dim,
                                                         weight_initializer=mx.init.Uniform(6 / math.sqrt(dim)))

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


if __name__ == '__main__':
    import random

    random.seed(10)

    batch_size = 3
    max_len = 10
    vec_dim = 5
    channel_size = 4
    length = [random.randint(1, max_len) for _ in range(128)]
    data = [[[[random.random() for _ in range(channel_size)] for _ in range(vec_dim)] for _ in range(max_len)] for l in
            length]

    data = mx.nd.array(data)
    print(data.shape)

    text_cnn = TextCNN(max_len, vec_dim, channel_size, 2)
    text_cnn.initialize()

    a = text_cnn(data)
    print(a.shape)
