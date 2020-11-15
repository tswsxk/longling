# coding: utf-8
# 2020/8/16 @ tongshiwei
import logging
import os

import mxnet as mx
from mxnet.initializer import Initializer, Xavier, Uniform, Normal


def net_initialize(
        net, model_ctx,
        initializer: (str, Initializer) = mx.init.Xavier(),
        select=None, logger=logging
):
    """
    初始化网络参数

    Notice
    ------
    The developer who modify this document should simultaneously modify the related function in glue

    """
    if isinstance(initializer, str):
        initializer = {
            "xaiver": Xavier(),
            "uniform": Uniform(),
            "normal": Normal()
        }[initializer]
    elif initializer is None or isinstance(initializer, Initializer):
        pass
    else:
        raise TypeError("initializer should be either str or Initializer, now is", type(initializer))

    logger.info("initializer: %s, select: %s, ctx: %s" % (initializer, select, model_ctx))
    net.collect_params(select).initialize(initializer, ctx=model_ctx)


def load_net(filename, net, ctx=mx.cpu(), allow_missing=False,
             ignore_extra=False):
    """
    Load the existing net parameters

    Parameters
    ----------
    filename: str
        The model file
    net: HybridBlock
        The network which has been initialized or
        loaded from the existed model
    ctx: Context or list of Context
            Defaults to ``mx.cpu()``.
    allow_missing: bool
    ignore_extra: bool

    Returns
    -------
    The initialized net
    """
    # 根据文件名装载已有的网络参数
    if not os.path.isfile(filename):
        raise FileExistsError("%s does not exist" % filename)
    net.load_parameters(filename, ctx, allow_missing=allow_missing,
                        ignore_extra=ignore_extra)
    return net
