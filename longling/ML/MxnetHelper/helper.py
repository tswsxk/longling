# coding: utf-8
# create by tongshiwei on 2019/4/7

import mxnet as mx
from mxnet import symbol, ndarray

__all__ = ["getF", "copy_net"]


def getF(input):
    r"""
    判断输入参数类型

    Parameters
    ----------
    input

    Returns
    -------

    """
    if isinstance(input, symbol.Symbol):
        return symbol
    elif isinstance(input, ndarray.NDArray):
        return ndarray
    else:
        raise TypeError(
            "the type of input should be either Symbol or NDArray, now is %s"
            % type(input)
        )


def copy_net(src_net, target_net, select=None):
    """
    复制网络

    Parameters
    ----------
    src_net
    target_net
    select

    Returns
    -------

    """
    src_params = src_net.collect_params(select=select)
    target_params = target_net.collect_params(select=select)

    for name, value in src_params._params.items():
        value = value.data()
        target_params._params[
            name.replace(src_params.prefix, target_params.prefix)
        ].set_data(value)


def get_fine_tune_model(symbol, label, arg_params, num_classes,
                        layer_name='flatten0'):
    """
    Only for static newwotk

    Do not use

    Parameters
    ----------
    symbol:
        the pretrained network symbol
    label:

    arg_params:
        the argument parameters of the pretrained model
    num_classes: int
        the number of classes for the fine-tune datasets
    layer_name: str
        the layer name before the last fully-connected layer
    Returns
    -------

    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name + '_output']
    net = mx.symbol.FullyConnected(
        data=net, num_hidden=num_classes, name='fc1'
    )
    net = mx.symbol.SoftmaxOutput(data=net, label=label, name='softmax')
    new_args = dict({k: arg_params[k] for k in arg_params if 'fc1' not in k})
    return net, new_args
