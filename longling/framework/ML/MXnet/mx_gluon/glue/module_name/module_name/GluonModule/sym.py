# coding: utf-8
# Copyright @tongshiwei

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


def net_viz(net, params, **kwargs):
    batch_size = params.batch_size
    model_dir = params.model_dir
    logger = kwargs.get('logger', params.logger if hasattr(params, 'logger') else logging)

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


def get_data_iter(params):
    # 在这里定义测试用伪数据流

    import random
    random.seed(10)

    batch_size = params.batch_size

    pass


if __name__ == '__main__':
    # # set parameters
    try:
        # for python module
        from .parameters import Parameters
    except (ImportError, SystemError):
        # for python script
        from parameters import Parameters

    params = Parameters()

    # # set batch size
    # batch_size = 128
    # params = Parameters(batch_size=batch_size)

    # generate sym
    net = NetName()

    # visualiztion check
    # params.view_tag = True
    # net_viz(net, params)
    #
    # net.initialize()
    #
    # # numerical check
    # datas = get_data_iter(params)

    from tqdm import tqdm

    bp_loss_f = lambda x, y: x - y

    # forward check
    # for data, label in tqdm(get_data_iter(params)):
    #     loss = bp_loss_f(net(data), label)

    # # train check
    # from mxnet import autograd
    # trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.01})
    # for epoch in range(0, 100):
    #     epoch_loss = 0
    #     for data, label, in get_data_iter(params):
    #         with autograd.record():
    #             pred_rs, _ = net(data)
    #             loss = bp_loss_f(pred_rs, label)
    #             epoch_loss += loss.asscalar()
    #             # epoch_loss += np.mean(loss.asnumpy())
    #             loss.backward()
    #         trainer.step(1)
    #     print(epoch_loss)
