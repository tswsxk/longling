# coding: utf-8
# Copyright @tongshiwei
"""
This file define the networks structure and
provide a simplest training and testing example.
"""
__all__ = ["NetName", "net_viz", "fit_f"]

import logging
import os

import mxnet as mx
from mxnet import gluon, nd, autograd

from longling.ML.MxnetHelper.toolkit.viz import plot_network, VizError

# set parameters
try:
    # for python module
    from .data import transform
    from .configuration import Configuration
except (ImportError, SystemError):
    # for python script
    from data import transform
    from configuration import Configuration


class NetName(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(NetName, self).__init__(**kwargs)

        with self.name_scope():
            pass

    def hybrid_forward(self, F, x, *args, **kwargs):
        pass


def net_viz(net, cfg, view_tag=False, **kwargs):
    """visualization check, only support pure static network"""
    batch_size = cfg.batch_size
    model_dir = cfg.model_dir
    logger = kwargs.get(
        'logger',
        cfg.logger if hasattr(cfg, 'logger') else logging
    )

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


def get_data_iter(cfg):
    def pesudo_data_generation():
        # 在这里定义测试用伪数据流
        import random
        random.seed(10)

        raw_data = [
            [random.random() for _ in range(5)]
            for _ in range(1000)
        ]

        return raw_data

    return transform(pesudo_data_generation(), cfg)


def fit_f(_data, bp_loss_f, loss_function, loss_monitor):
    data, label = _data
    # todo modify the input to net
    output = net(data)
    bp_loss = None
    for name, func in loss_function.items():
        # todo modify the input to func
        loss = func(output, label)
        if name in bp_loss_f:
            bp_loss = loss
        loss_value = nd.mean(loss).asscalar()
        if loss_monitor:
            loss_monitor.update(name, loss_value)
    return bp_loss


def numerical_check(net, cfg):
    net.initialize()

    datas = get_data_iter(cfg)

    bp_loss_f = {"L2Loss": gluon.loss.L2Loss}
    loss_function = {}
    loss_function.update(bp_loss_f)
    from longling.ML.toolkit.monitor import MovingLoss
    from longling.ML.MxnetHelper.glue import module

    loss_monitor = MovingLoss(loss_function)

    # train check
    trainer = module.Module.get_trainer(
        net, optimizer=cfg.optimizer,
        optimizer_params=cfg.optimizer_params,
        select=cfg.train_select
    )

    for epoch in range(0, 100):
        epoch_loss = 0
        for _data in datas:
            with autograd.record():
                fit_f(_data, bp_loss_f, loss_function, loss_monitor)
            trainer.step(cfg.batch_size)
        print(epoch_loss)


if __name__ == '__main__':
    cfg = Configuration()

    # generate sym
    net = NetName()

    # # visualiztion check
    # net_viz(net, params, False)

    numerical_check(net, cfg)
