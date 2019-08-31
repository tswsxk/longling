# coding: utf-8
# create by tongshiwei on 2019/4/12
import os

__all__ = ["Module"]
import torch

from ..toolkit.optimizer import get_trainer


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
    def load_net(filename, net: torch.nn.Module, ctx="cpu", allow_missing=False,
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
        return torch.load(filename, map_location=lambda s, loc: s)

    def load(self, net, epoch, ctx="cpu", allow_missing=False,
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
    def get_trainer(net, optimizer='sgd', optimizer_params=None, select=None):
        """把优化器安装到网络上"""
        trainer = get_trainer(net, optimizer, optimizer_params, select)
        return trainer

    @staticmethod
    def save_params(filename, net: torch.nn.Module):
        torch.save(net.state_dict(), filename)
