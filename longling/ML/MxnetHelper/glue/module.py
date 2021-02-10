# coding: utf-8
# create by tongshiwei on 2019/4/12
import functools
import logging

import mxnet as mx
from longling.ML import DL
from longling.ML.MxnetHelper.toolkit import get_trainer, load_net, save_params

__all__ = ["Module"]


class Module(DL.Module):
    def fit_f(self, *args, **kwargs):
        raise NotImplementedError

    def sym_gen(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, net, epoch, ctx=mx.cpu(), allow_missing=False,
             ignore_extra=False):
        """"
        Load the existing net parameters

        Parameters
        ----------
        net: HybridBlock
            The network which has been initialized
            or loaded from the existed model
        epoch: str or int
            The epoch which specify the model
        ctx: Context or list of Context
                Defaults to ``mx.cpu()``.
        allow_missing: bool
        ignore_extra: bool

        Returns
        -------
        The initialized net
        """
        # 根据起始轮次装载已有的网络参数
        filename = self.epoch_params_filepath(epoch)
        return load_net(
            filename, net, ctx, allow_missing=allow_missing,
            ignore_extra=ignore_extra
        )

    def epoch_params_filepath(self, epoch):
        raise NotImplementedError

    def net_initialize(self, *args, **kwargs):
        """初始化网络参数"""
        raise NotImplementedError

    @staticmethod
    @functools.wraps(get_trainer)
    def get_trainer(net, optimizer='sgd', optimizer_params=None, lr_params=None, select=None, logger=logging,
                    *args, **kwargs):
        return get_trainer(
            net=net, optimizer=optimizer, optimizer_params=optimizer_params,
            lr_params=lr_params, select=select, logger=logger, *args, **kwargs
        )

    @staticmethod
    @functools.wraps(save_params)
    def save_params(filename, net, select):
        save_params(filename=filename, net=net, select=select)

    @staticmethod
    def eval(*args, **kwargs):
        raise NotImplementedError
