# coding:utf-8
# created by tongshiwei on 2018/8/2

from mxnet import symbol, ndarray

from mxnet.gluon.parameter import tensor_types


def getF(input):
    if isinstance(input, symbol.Symbol):
        return symbol
    elif isinstance(input, ndarray.NDArray):
        return ndarray
