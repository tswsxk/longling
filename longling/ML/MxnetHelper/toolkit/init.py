# coding: utf-8
# 2020/8/16 @ tongshiwei
import logging
import os

import mxnet as mx
from mxnet.initializer import Initializer, Xavier, Uniform, Normal


def net_initialize(
        net, model_ctx,
        initializer: (str, Initializer, dict, list) = mx.init.Xavier(),
        select=None, logger=logging, verbose=False, force_reinit=False
):
    """
    初始化网络参数

    Parameters
    ----------
    net
    model_ctx: mx.cpu or mx.gpu
    initializer: str, Initializer, dict or list, tuple
    select
    logger
    verbose : bool, default False
        Whether to verbosely print out details on initialization.
    force_reinit : bool, default False
        Whether to force re-initialization if parameter is already initialized.
    Notes
    ------
    The developer who modify this document should simultaneously modify the related function in glue

    Examples
    --------
    >>> import mxnet as mx
    >>> from mxnet import gluon
    >>> emb = gluon.nn.Embedding(2, 3)
    >>> net_initialize(emb, mx.cpu())
    >>> emb.weight.data()
    <BLANKLINE>
    [[0.10694504 0.2034123  0.4714563 ]
     [0.7542485  0.2251432  0.7842196 ]]
    <NDArray 2x3 @cpu(0)>
    >>> emb1 = gluon.nn.Embedding(2, 3)
    >>> net_initialize(emb1, mx.cpu(), initializer=mx.init.Xavier())
    >>> emb1.weight.data()
    <BLANKLINE>
    [[ 0.09833419  0.76079047 -0.16726398]
     [ 0.27071452  0.319638   -0.25330698]]
    <NDArray 2x3 @cpu(0)>
    >>> class EmbNet(gluon.nn.HybridBlock):
    ...     def __init__(self, prefix=None, params=None):
    ...         super(EmbNet, self).__init__(prefix, params)
    ...         with self.name_scope():
    ...             self.emb = gluon.nn.Embedding(2, 3)
    ...             self.linear = gluon.nn.Dense(4)
    ...     def hybrid_forward(self, F, x, *args, **kwargs):
    ...         return self.linear(self.emb(x))
    >>> net = EmbNet()
    >>> from longling.ML.DL import BLOCK_EMBEDDING
    >>> net_initialize(net, mx.cpu(), initializer={BLOCK_EMBEDDING: "xaiver", ".*embedding": "uniform"})
    >>> net(mx.nd.array([0, 1]))
    <BLANKLINE>
    [[ 0.03268543 -0.00860071  0.04774952  0.00056277]
     [-0.00648303 -0.03121923 -0.04578817 -0.08059631]]
    <NDArray 2x4 @cpu(0)>
    >>> net1 = EmbNet()
    >>> net_initialize(net1, mx.cpu(), initializer=["xaiver", "uniform"], select=[BLOCK_EMBEDDING, ".*embedding"])
    >>> net1(mx.nd.array([0, 1]))  # doctest: +ELLIPSIS
    <BLANKLINE>
    [[-0.0896... -0.0179... -0.0156... -0.0136...]
     [ 0.0033...  0.0255...  0.0111...  0.0446...]]
    <NDArray 2x4 @cpu(0)>
    >>> net_initialize(net1, mx.cpu(), initializer=[(BLOCK_EMBEDDING, "xaiver"), (".*embedding", "uniform")],
    ...     force_reinit=True)
    >>> net1(mx.nd.array([0, 1]))  # doctest: +ELLIPSIS
    <BLANKLINE>
    [[ 0.0153...  0.0266... -0.0466...  0.0291...]
     [-0.0362...  0.0063...  0.0227... -0.0212...]]
    <NDArray 2x4 @cpu(0)>
    """
    if isinstance(initializer, str):
        initializer = {
            "xaiver": Xavier(),
            "uniform": Uniform(),
            "normal": Normal()
        }[initializer]
    elif isinstance(initializer, dict):
        for _select, _initializer in initializer.items():
            net_initialize(net, model_ctx=model_ctx, initializer=_initializer, select=_select, logger=logger,
                           verbose=verbose, force_reinit=force_reinit)
        return
    elif isinstance(initializer, (list, tuple)):
        if select is not None:
            assert len(select) == len(initializer)
            for _select, _initializer in zip(select, initializer):
                net_initialize(net, model_ctx=model_ctx, initializer=_initializer, select=_select, logger=logger,
                               verbose=verbose, force_reinit=force_reinit)
        else:
            for _select, _initializer in initializer:
                net_initialize(net, model_ctx=model_ctx, initializer=_initializer, select=_select, logger=logger,
                               verbose=verbose, force_reinit=force_reinit)
        return
    elif initializer is None or isinstance(initializer, Initializer):
        pass
    else:
        raise TypeError("initializer should be either str or Initializer, now is", type(initializer))

    logger.info("initializer: %s, select: %s, ctx: %s" % (initializer, select, model_ctx))
    net.collect_params(select).initialize(initializer, ctx=model_ctx, verbose=verbose, force_reinit=force_reinit)


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
