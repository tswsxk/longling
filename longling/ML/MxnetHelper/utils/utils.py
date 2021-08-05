# coding: utf-8
# create by tongshiwei on 2019/4/7

from mxnet import symbol, ndarray

__all__ = ["getF", "copy_net"]


def getF(input_object: (symbol.Symbol, ndarray.NDArray)) -> (symbol, ndarray):
    r"""
    判断输入参数类型

    Parameters
    ----------
    input_object

    Returns
    -------
    The type of input object

    Examples
    --------
    >>> import mxnet as mx
    >>> getF(mx.nd.zeros((3, 3)))   # doctest: +ELLIPSIS
    <module 'mxnet.ndarray'...
    >>> getF(mx.symbol.zeros((3, 3)))   # doctest: +ELLIPSIS
    <module 'mxnet.symbol'...
    """
    if isinstance(input_object, symbol.Symbol):
        return symbol
    elif isinstance(input_object, ndarray.NDArray):
        return ndarray
    else:
        raise TypeError(
            "the type of input should be either Symbol or NDArray, now is %s"
            % type(input_object)
        )


def copy_net(src_net, target_net, select=None):
    """
    复制网络

    Parameters
    ----------
    src_net
    target_net
    select

    Examples
    --------
    >>> import mxnet as mx
    >>> from mxnet import gluon
    >>> emb1 = gluon.nn.Embedding(2, 3)
    >>> emb1.initialize()
    >>> emb1.weight.set_data(mx.nd.ones((2, 3)))
    >>> emb1.weight.data()
    <BLANKLINE>
    [[1. 1. 1.]
     [1. 1. 1.]]
    <NDArray 2x3 @cpu(0)>
    >>> emb2 = gluon.nn.Embedding(2, 3)
    >>> emb2.initialize()
    >>> emb2.weight.set_data(mx.nd.zeros((2, 3)))
    >>> emb2.weight.data()
    <BLANKLINE>
    [[0. 0. 0.]
     [0. 0. 0.]]
    <NDArray 2x3 @cpu(0)>
    >>> copy_net(emb1, emb2)
    >>> emb2.weight.data()
    <BLANKLINE>
    [[1. 1. 1.]
     [1. 1. 1.]]
    <NDArray 2x3 @cpu(0)>
    """
    src_params = src_net.collect_params(select=select)
    target_params = target_net.collect_params(select=select)

    for name, value in src_params._params.items():
        value = value.data()
        target_params._params[
            name.replace(src_params.prefix, target_params.prefix)
        ].set_data(value)
