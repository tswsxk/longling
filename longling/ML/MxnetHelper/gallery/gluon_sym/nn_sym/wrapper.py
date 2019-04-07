# coding:utf-8
# created by tongshiwei on 2018/7/31

from mxnet import ndarray, symbol


class PVWeight(object):
    def __init__(self, params, var):
        self.params = params
        self.var = var

    def __call__(self, F):
        if F is ndarray:
            return self.params.data()
        elif F is symbol:
            return self.var
        else:
            return None