# coding: utf-8
# created by tongshiwei on 18-2-6
import math

import mxnet as mx
from mxnet import gluon


class TextCNN(gluon.HybridBlock):
    def __init__(self, sentence_size, vec_size, channel_size=None, num_output=2,
                 filter_list=(1, 2, 3, 4), num_filters=60,
                 dropout=0.0, batch_norm=True, activation="tanh",
                 num_highway=1,
                 highway_layer_activation='relu', highway_trans_layer_activation=None,
                 pool_type='max',
                 **kwargs):
        """
        TextCNN 模型，for 2D and 3D

        Parameters
        ----------
        sentence_size: int
            句子长度
        vec_size: int
            特征向量维度
        channel_size: int or None
            if None, use Text2D
            int 则为 channel（特征向量二维维度）
        num_output: int
            输出维度
        filter_list: Iterable
            The output dimension for each convolutional layer according to the filter sizes,
            which are the number of the filters learned by the layers.
        num_filters: int or Iterable
            The size of each convolutional layer, int or Iterable.
            When Iterable, len(filter_list) equals to the number of convolutional layers.
        dropout: float
        batch_norm: bool
        activation: str
        num_highway: int
        highway_layer_activation: str
        highway_trans_layer_activation: str or None
        pool_type: str
        kwargs
        """
        super(TextCNN, self).__init__(**kwargs)
        self.sentence_size = sentence_size
        self.vec_size = vec_size
        self.channel_size = channel_size
        self.num_output = num_output
        self.filter_list = filter_list
        if isinstance(num_filters, int):
            self.num_filters = [num_filters] * len(self.filter_list)
        assert len(self.filter_list) == len(self.num_filters)
        self.batch_norm = batch_norm
        self.num_highway = num_highway
        self.activation = activation
        self.highway_layer_activation = highway_layer_activation
        self.highway_trans_layer_activation = highway_trans_layer_activation if highway_trans_layer_activation \
            else highway_layer_activation

        self.conv = [0] * len(self.filter_list)
        self.pool = [0] * len(self.filter_list)
        self.bn = [0] * len(self.filter_list)

        pool2d = gluon.nn.MaxPool2D if pool_type == "max" else gluon.nn.AvgPool2D
        pool3d = gluon.nn.MaxPool3D if pool_type == "max" else gluon.nn.AvgPool3D

        with self.name_scope():
            for i, (filter_size, num_filter) in enumerate(zip(self.filter_list, self.num_filters)):
                conv = gluon.nn.Conv2D(num_filter, kernel_size=(filter_size, self.vec_size),
                                       activation=self.activation) if not self.channel_size else gluon.nn.Conv3D(
                    num_filter, kernel_size=(filter_size, self.vec_size, self.channel_size), activation=activation)
                setattr(self, "conv%s" % i, conv)

                pool = pool2d(pool_size=(self.sentence_size - filter_size + 1, 1),
                              strides=(1, 1)) if not self.channel_size else pool3d(
                    pool_size=(self.sentence_size - filter_size + 1, 1, 1), strides=(1, 1, 1))
                setattr(self, "pool%s" % i, pool)

                if self.batch_norm:
                    setattr(self, "bn%s" % i, gluon.nn.BatchNorm())

            if self.num_highway:
                dim = sum(self.num_filters)
                for i in range(self.num_highway):
                    setattr(self, "highway%s" % i,
                            HighwayCell(dim, highway_layer_activation=self.highway_layer_activation,
                                        highway_trans_layer_activation=self.highway_trans_layer_activation))

            self.dropout = gluon.nn.Dropout(dropout)

            self.fc = gluon.nn.Dense(num_output)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = F.expand_dims(x, axis=1)

        pooled_outputs = []
        for i, _ in enumerate(self.filter_list):
            convi = getattr(self, "conv%s" % i)(x)
            pooli = getattr(self, "pool%s" % i)(convi)
            if self.batch_norm:
                pooli = getattr(self, "bn%s" % i)(pooli)
            pooled_outputs.append(pooli)

        total_filters = sum(self.num_filters)
        concat = F.Concat(dim=1, *pooled_outputs)
        h_pool = F.Reshape(data=concat, shape=(0, total_filters))
        if self.num_highway:
            for i in range(self.num_highway):
                h_pool = getattr(self, "highway%s" % i)(h_pool)

        h_drop = self.dropout(h_pool)

        fc = self.fc(h_drop)

        return fc


class HighwayCell(gluon.HybridBlock):
    def __init__(self, dim, highway_layer_activation='relu', highway_trans_layer_activation=None, prefix=None,
                 params=None):
        super(HighwayCell, self).__init__(prefix, params)
        highway_trans_layer_activation = highway_trans_layer_activation if highway_trans_layer_activation \
            else highway_layer_activation
        with self.name_scope():
            self.high_fc = gluon.nn.Dense(dim, activation=highway_layer_activation)
            self.high_trans_fc = gluon.nn.Dense(dim, activation=highway_trans_layer_activation)

    def hybrid_forward(self, F, x, *args, **kwargs):
        high_fc = self.high_fc(x)
        high_trans_fc = self.high_trans_fc(x)
        return high_fc * high_trans_fc + x * (1 - high_trans_fc)


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
