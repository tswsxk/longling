# coding: utf-8
# create by tongshiwei on 2019-9-1

from mxnet import gluon


# todo: define your network symbol here
class NetName(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(NetName, self).__init__(kwargs.get("prefix"), kwargs.get("params"))

        with self.name_scope():
            pass

    def hybrid_forward(self, F, x, *args, **kwargs):
        pass


BP_LOSS_F = {"L2Loss": gluon.loss.L2Loss()}
