# coding: utf-8
# create by tongshiwei on 2019/4/12
import logging
import re

import mxnet as mx
from mxnet import gluon, nd

from longling.ML import DL
from longling.ML.MxnetHelper.toolkit.init import load_net

__all__ = ["Module"]


class Module(DL.Module):
    def fit_f(self):
        raise NotImplementedError

    def sym_gen(self):
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
        filename = self.epoch_params_filename(epoch)
        return load_net(
            filename, net, ctx, allow_missing=allow_missing,
            ignore_extra=ignore_extra
        )

    def epoch_params_filename(self, epoch):
        raise NotImplementedError

    def net_initialize(self, *args, **kwargs):
        """初始化网络参数"""
        raise NotImplementedError

    @staticmethod
    def get_trainer(net, optimizer='sgd', optimizer_params=None, lr_params=None, select=None, logger=logging):
        """把优化器安装到网络上"""
        if lr_params:
            if "update_params" in lr_params and len(lr_params) == 1:
                pass
            else:
                from longling.ML.MxnetHelper.toolkit import get_lr_scheduler
                optimizer_params["lr_scheduler"] = get_lr_scheduler(logger=logger, **lr_params)

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


def module_wrapper(
        module_cls: type(Module),
        net_gen_func,
        fit_func, configuration_cls=None):
    _select = configuration_cls.train_select if configuration_cls is not None else None

    class MetaModule(module_cls):
        @property
        def sym_gen(self):
            return net_gen_func

        @property
        def fit_func(self):
            return fit_func

        @staticmethod
        def get_trainer(net, optimizer='sgd', optimizer_params=None, lr_params=None, select=_select):
            """把优化器安装到网络上"""
            return module_cls.get_trainer(
                net, optimizer, optimizer_params, lr_params, select
            )

    return MetaModule
