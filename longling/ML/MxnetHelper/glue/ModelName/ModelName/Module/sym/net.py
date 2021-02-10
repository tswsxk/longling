# coding: utf-8
# create by tongshiwei on 2019-9-1

__all__ = ["get_net", "get_loss"]

# todo: define your network symbol and back propagation loss function

from mxnet import gluon

from longling.ML.MxnetHelper.toolkit import loss_dict2tmt_mx_loss


def get_net(*args, **kwargs):
    return NetName(*args, **kwargs)


def get_loss(**kwargs):
    return loss_dict2tmt_mx_loss({
        "L2Loss": gluon.loss.L2Loss(**kwargs)
    })


class NetName(gluon.HybridBlock):
    def __init__(self, prefix=None, params=None):
        super(NetName, self).__init__(prefix, params)

        with self.name_scope():
            pass

    def hybrid_forward(self, F, x, *args, **kwargs):
        pass
