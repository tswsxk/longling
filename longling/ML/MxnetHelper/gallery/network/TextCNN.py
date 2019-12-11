# coding: utf-8
# create by tongshiwei on 2019/4/7

from mxnet import gluon

__all__ = ["TextCNN"]


class TextCNN(gluon.HybridBlock):
    def __init__(self, sentence_size, vec_size, channel_size=None,
                 num_output=2, filter_list=(1, 2, 3, 4), num_filters=60,
                 dropout=0.0, batch_norm=True, activation="tanh",
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
            The output dimension for each convolutional layer according
            to the filter sizes,
            which are the number of the filters learned by the layers.
        num_filters: int or Iterable
            The size of each convolutional layer, int or Iterable.
            When Iterable, len(filter_list) equals to the number of
            convolutional layers.
        dropout: float
        batch_norm: bool
        activation: str
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
        self.activation = activation

        self.conv = [0] * len(self.filter_list)
        self.pool = [0] * len(self.filter_list)
        self.bn = [0] * len(self.filter_list)

        pool2d = gluon.nn.MaxPool2D if pool_type == "max" \
            else gluon.nn.AvgPool2D

        pool3d = gluon.nn.MaxPool3D if pool_type == "max" \
            else gluon.nn.AvgPool3D

        with self.name_scope():
            for i, (filter_size, num_filter) in enumerate(
                    zip(self.filter_list, self.num_filters)):
                conv = gluon.nn.Conv2D(
                    num_filter,
                    kernel_size=(filter_size, self.vec_size),
                    activation=self.activation
                ) if not self.channel_size else gluon.nn.Conv3D(
                    num_filter,
                    kernel_size=(filter_size, self.vec_size, self.channel_size),
                    activation=activation
                )
                setattr(self, "conv%s" % i, conv)

                pool = pool2d(
                    pool_size=(self.sentence_size - filter_size + 1, 1),
                    strides=(1, 1)) if not self.channel_size else pool3d(
                    pool_size=(self.sentence_size - filter_size + 1, 1, 1),
                    strides=(1, 1, 1))
                setattr(self, "pool%s" % i, pool)

                if self.batch_norm:
                    setattr(self, "bn%s" % i, gluon.nn.BatchNorm())

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
        h_drop = self.dropout(h_pool)

        fc = self.fc(h_drop)

        return fc
