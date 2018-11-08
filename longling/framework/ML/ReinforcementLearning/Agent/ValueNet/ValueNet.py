# coding:utf-8
# created by tongshiwei on 2018/11/8

from mxnet import gluon


class NetSym(gluon.HybridBlock):
    def __init__(self, prefix=None, params=None):
        super(NetSym, self).__init__(prefix=prefix, params=params)

        with self.name_scope():
            pass

    def hybrid_forward(self, F, x, *args, **kwargs):
        pass


class ValueNet(object):
    def __init__(self):
        pass
