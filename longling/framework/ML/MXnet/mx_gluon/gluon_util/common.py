# coding:utf-8
# created by tongshiwei on 2018/8/2

from mxnet import symbol, ndarray

from mxnet.gluon.parameter import tensor_types


def getF(input):
    if isinstance(input, symbol.Symbol):
        return symbol
    elif isinstance(input, ndarray.NDArray):
        return ndarray


def copy_net(src_net, target_net, select=None):
    src_params = src_net.collect_params(select=select)
    target_params = target_net.collect_params(select=select)

    for name, value in src_params._params.items():
        value = value.data()
        target_params._params[name.replace(src_params.prefix, target_params.prefix)].set_data(value)
