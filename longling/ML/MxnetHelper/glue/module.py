# coding: utf-8
# create by tongshiwei on 2019/4/12
import os
import re

import mxnet as mx
from mxnet import gluon, nd

__all__ = ["Module"]


class Module(object):
    def __init__(self, parameters):
        self.cfg = parameters

    def __str__(self):
        """
        显示模块参数
        Display the necessary params of this Module

        Returns
        -------

        """
        string = ["Params"]
        for k, v in vars(self.cfg).items():
            string.append("%s: %s" % (k, v))
        return "\n".join(string)

    @staticmethod
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
            raise FileExistsError
        net.load_parameters(filename, ctx, allow_missing=allow_missing,
                            ignore_extra=ignore_extra)
        return net

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
        filename = self.epoch_params_filename(epoch)
        return self.load_net(
            filename, net, ctx, allow_missing=allow_missing,
            ignore_extra=ignore_extra
        )

    def epoch_params_filename(self, epoch):
        raise NotImplementedError

    @staticmethod
    def net_initialize(
            net, model_ctx, initializer=mx.init.Normal(sigma=.1), select=None
    ):
        """初始化网络参数"""
        net.collect_params(select).initialize(initializer, ctx=model_ctx)

    @staticmethod
    def get_trainer(net, optimizer='sgd', optimizer_params=None, select=None):
        """把优化器安装到网络上"""
        trainer = gluon.Trainer(
            net.collect_params(select), optimizer, optimizer_params
        )
        return trainer

    @staticmethod
    def save_params(filename, net, select):
        params = net._collect_params_with_prefix()
        if select:
            pattern = re.compile(select)
            params = {name: value for name, value in params.items() if
                      pattern.match(name)}
        arg_dict = {key: val._reduce() for key, val in params.items()}
        nd.save(filename, arg_dict)
