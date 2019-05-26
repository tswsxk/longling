# coding: utf-8
# Copyright @tongshiwei
"""
This file define the networks structure and
provide a simplest training and testing example.
"""

import logging
import os

import mxnet as mx
from mxnet import gluon, nd, autograd
from tqdm import tqdm

from longling.ML.MxnetHelper.toolkit.ctx import split_and_load
from longling.ML.MxnetHelper.toolkit.viz import plot_network, VizError

# set parameters
try:
    # for python module
    from .etl import transform
    from .configuration import Configuration
except (ImportError, SystemError):
    # for python script
    from etl import transform
    from configuration import Configuration

__all__ = ["NetName", "net_viz", "fit_f", "BP_LOSS_F", "eval_f"]


# todo: define your network symbol here
class NetName(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(NetName, self).__init__(**kwargs)

        with self.name_scope():
            pass

    def hybrid_forward(self, F, x, *args, **kwargs):
        pass


# todo: optional, visualize the network
def net_viz(_net, _cfg, view_tag=False, **kwargs):
    """visualization check, only support pure static network"""
    batch_size = _cfg.batch_size
    model_dir = _cfg.model_dir
    logger = kwargs.get(
        'logger',
        _cfg.logger if hasattr(_cfg, 'logger') else logging
    )

    try:
        viz_dir = os.path.join(model_dir, "plot/network")
        logger.info("visualization: file in %s" % viz_dir)
        from copy import deepcopy

        viz_net = deepcopy(_net)
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


# todo:
def get_data_iter(_cfg):
    def pseudo_data_generation():
        # 在这里定义测试用伪数据流
        import random
        random.seed(10)

        raw_data = [
            [random.random() for _ in range(5)]
            for _ in range(1000)
        ]

        return raw_data

    return transform(pseudo_data_generation(), _cfg)


def fit_f(_net, _data, bp_loss_f, loss_function, loss_monitor):
    data, label = _data
    # todo modify the input to net
    output = _net(data)
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


def eval_f(_net, test_data, ctx=mx.cpu()):
    ground_truth = []
    prediction = []

    def evaluation_function(y_true, y_pred):
        return 0

    for batch_data in tqdm(test_data, "evaluating"):
        ctx_data = split_and_load(
            ctx, *batch_data,
            even_split=False
        )
        for (data, label) in ctx_data:
            output = _net(data)
            pred = output
            ground_truth.extend(label.asnumpy().tolist())
            prediction.extend(pred.asnumpy().tolist())

    return {
        "evaluation_name": evaluation_function(ground_truth, prediction)
    }


BP_LOSS_F = {"L2Loss": gluon.loss.L2Loss}


def numerical_check(_net, _cfg):
    net.initialize()

    datas = get_data_iter(_cfg)

    bp_loss_f = BP_LOSS_F
    loss_function = {}
    loss_function.update(bp_loss_f)
    from longling.ML.toolkit.monitor import MovingLoss
    from longling.ML.MxnetHelper.glue import module

    loss_monitor = MovingLoss(loss_function)

    # train check
    trainer = module.Module.get_trainer(
        _net, optimizer=_cfg.optimizer,
        optimizer_params=_cfg.optimizer_params,
        select=_cfg.train_select
    )

    for epoch in range(0, 100):
        for _data in tqdm(datas):
            with autograd.record():
                bp_loss = fit_f(
                    _net, _data, bp_loss_f, loss_function, loss_monitor
                )
            assert bp_loss is not None
            bp_loss.backward()
            trainer.step(_cfg.batch_size)
        print("epoch-%d: %s" % (epoch, list(loss_monitor.items())))


if __name__ == '__main__':
    cfg = Configuration()

    # generate sym
    net = NetName()

    # # visualiztion check
    # net_viz(net, cfg, False)

    # # numerical check
    # numerical_check(net, cfg)
