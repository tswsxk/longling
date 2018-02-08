# coding: utf-8
# created by tongshiwei on 18-2-6

from mxnet import gluon


# class TextCNN(gluon.HybridBlock):
#     def __init__(self, units, activation=None, use_bias=True, flatten=True,
#                  weight_initializer=None, bias_initializer='zeros',
#                  in_units=0, **kwargs):
#         super(TextCNN, self).__init__(**kwargs)
#         self._flatten = flatten
#         with self.name_scope():
#             self._units = units
#             self._in_units = in_units
#             self.weight = self.params.get('weight', shape=(units, in_units),
#                                           init=weight_initializer,
#                                           allow_deferred_init=True)
#             if use_bias:
#                 self.bias = self.params.get('bias', shape=(units,),
#                                             init=bias_initializer,
#                                             allow_deferred_init=True)
#             else:
#                 self.bias = None
#             if activation is not None:
#                 self.act = gluon.nn.Activation(activation, prefix=activation+'_')
#             else:
#                 self.act = None
#
#     def hybrid_forward(self, F, x, weight, bias=None):
#         act = F.FullyConnected(x, weight, bias, no_bias=bias is None, num_hidden=self._units,
#                                flatten=self._flatten, name='fwd')
#         if self.act is not None:
#             act = self.act(act)
#         return act
#
#     def __repr__(self):
#         s = '{name}({layout}, {act})'
#         shape = self.weight.shape
#         return s.format(name=self.__class__.__name__,
#                         act=self.act if self.act else 'linear',
#                         layout='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]))


class TextCNN(gluon.HybridBlock):
    def __init__(self, sentence_size, vec_size, vocab_size=None, num_output=2, filter_list=[1, 2, 3, 4], num_filter=60,
                 dropout=0.0, batch_norms=0, highway=False, activation="relu", fix_embedding=True,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.sentence_size = sentence_size
        self.vec_size = vec_size
        self.num_output = num_output
        self.filter_list = filter_list
        self.num_filter = num_filter
        self.dropout_p = dropout
        self.batch_norms = batch_norms
        self.highway = highway
        self.activation = activation

        # self.conv = [0] * len(self.filter_list)
        # self.pool = [0] * len(self.filter_list)
        # self.bn = [0] * len(self.filter_list)
        #
        # self.high_fc = None
        # self.high_trans_fc = None
        #
        # self.dropout = None

        create_var = locals()
        self.fix_embedding = fix_embedding
        self.vocab_size = vocab_size

        with self.name_scope():
            if vocab_size is not None:
                self.embedding = gluon.nn.Embedding(vocab_size, vec_size)

            for i, filter_size in enumerate(self.filter_list):
                conv = gluon.nn.Conv2D(self.num_filter, kernel_size=(filter_size, self.vec_size),
                                       activation=self.activation)
                setattr(self, "conv%s" % i, conv)

                pool = gluon.nn.MaxPool2D(pool_size=(self.sentence_size - filter_size + 1, 1), strides=(1, 1))
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

        if self.embedding is not None:
            x = self.embedding(x)
            if self.fix_embedding:
                x = F.BlockGrad(x)

        pooled_outputs = []
        for i, _ in enumerate(self.filter_list):
            convi = eval("self.conv%s" % i)(x)
            pooli = eval("self.pool%s" % i)(convi)
            if self.batch_norms > 0:
                pooli = eval("self.bn%s" % i)(pooli)
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
