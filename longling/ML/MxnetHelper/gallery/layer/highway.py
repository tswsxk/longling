# coding: utf-8
# create by tongshiwei on 2019/4/7

from mxnet import gluon

__all__ = ["HighwayCell"]


class HighwayCell(gluon.HybridBlock):
    def __init__(self, dim, highway_layer_activation='relu',
                 highway_trans_layer_activation=None, prefix=None,
                 params=None):
        super(HighwayCell, self).__init__(prefix, params)
        highway_trans_layer_activation = highway_trans_layer_activation \
            if highway_trans_layer_activation else highway_layer_activation
        with self.name_scope():
            self.high_fc = gluon.nn.Dense(
                dim,
                activation=highway_layer_activation
            )
            self.high_trans_fc = gluon.nn.Dense(
                dim,
                activation=highway_trans_layer_activation
            )

    def hybrid_forward(self, F, x, *args, **kwargs):
        high_fc = self.high_fc(x)
        high_trans_fc = self.high_trans_fc(x)
        return high_fc * high_trans_fc + x * (1 - high_trans_fc)
