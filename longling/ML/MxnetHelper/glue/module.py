# coding: utf-8
# create by tongshiwei on 2019/4/12
import functools
import logging

import mxnet as mx

from longling.ML import DL
from longling.ML.MxnetHelper.toolkit import get_trainer, load_net, save_params

__all__ = ["Module", "module_wrapper"]


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


def module_wrapper(
        module_cls: type(Module),
        net_gen_func=None,
        fit_func=None,
        net_init_func=None):
    """

    Parameters
    ----------
    module_cls
    net_gen_func
    fit_func
    net_init_func

    Returns
    -------

    """
    net_gen_func = module_cls.sym_gen if net_gen_func is None else net_gen_func
    fit_func = module_cls.fit_f if fit_func is None else fit_func
    net_init_func = module_cls.net_initialize if net_init_func is None else net_init_func

    class MetaModule(module_cls):
        @functools.wraps(net_gen_func)
        def sym_gen(self, *args, **kwargs):
            return net_gen_func(*args, **kwargs)

        @functools.wraps(fit_func)
        def fit_f(self, *args, **kwargs):
            return fit_func(*args, **kwargs)

        @functools.wraps(net_init_func)
        def net_initialize(self, *args, **kwargs):
            return net_init_func(*args, **kwargs)

    return MetaModule
