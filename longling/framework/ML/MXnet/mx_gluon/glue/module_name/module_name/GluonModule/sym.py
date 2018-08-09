# coding: utf-8
# create by tongshiwei on 2018/8/5

import logging
import os

import mxnet as mx
from mxnet import gluon

from longling.framework.ML.MXnet.viz import plot_network, VizError


class NetName(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(NetName, self).__init__(**kwargs)

        with self.name_scope():
            pass

    def hybrid_forward(self, F, x, *args, **kwargs):
        pass


def net_viz(net, params, logger=logging):
    batch_size = params.batch_size
    model_dir = params.model_dir

    try:
        view_tag = params.view_tag
    except AttributeError:
        view_tag = True

    try:
        viz_dir = os.path.abspath(model_dir + "plot/network")
        logger.info("visualization: file in %s" % viz_dir)
        from copy import deepcopy

        viz_net = deepcopy(net)
        viz_shape = {'data': (batch_size,) + (1,)}
        x = mx.sym.var("data")
        sym = viz_net(x)
        plot_network(
            nn_symbol=sym,
            save_path=viz_dir,
            shape=viz_shape,
            node_attrs={"fixedsize": "false"},
            view=view_tag
        )
    except VizError as e:
        logger.error("error happen in visualization, aborted")
        logger.error(e)


# def get_data_iter():
#     pass

# if __name__ == '__main__':
    # set parameters
    # try:
    #     from .parameters import Parameters
    # except (ImportError, SystemError):
    #     from parameters import Parameters
    #
    # batch_size = 128
    # params = Parameters(batch_size=batch_size)
    #
    # generate sym
    # net = NetName()
    #
    # visualiztion check
    # net_viz(net, params, params.logger)
    #
    # numerical check
    # data = get_data_iter()
    # net()
