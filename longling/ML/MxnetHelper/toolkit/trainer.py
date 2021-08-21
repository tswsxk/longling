# coding: utf-8
# create by tongshiwei on 2020-11-12

import logging

from copy import deepcopy
from mxnet import gluon


def get_trainer(net, optimizer='sgd', optimizer_params=None, lr_params=None, select=None, logger=logging,
                *args, **kwargs):
    """
    把优化器安装到网络上

    New in version 1.3.16
    """
    if lr_params:
        if "update_params" in lr_params and len(lr_params) == 1:
            pass
        else:
            from longling.ML.MxnetHelper.toolkit import get_lr_scheduler
            optimizer_params = deepcopy(optimizer_params)
            optimizer_params["lr_scheduler"] = get_lr_scheduler(logger=logger, **lr_params)

    trainer = gluon.Trainer(
        net.collect_params(select), optimizer, optimizer_params
    )
    return trainer
