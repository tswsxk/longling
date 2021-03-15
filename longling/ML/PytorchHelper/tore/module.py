# coding: utf-8
# create by tongshiwei on 2019/4/12
import os

__all__ = ["Module"]

import torch

from ..toolkit.trainer import get_trainer
from ..toolkit.parallel import set_device
from longling.ML import DL


class Module(DL.Module):
    @staticmethod
    def load_net(filename):
        """
        Load the existing net parameters

        Parameters
        ----------
        filename: str
            The model file
        Returns
        -------
        The initialized net
        """
        # 根据文件名装载已有的网络参数
        if not os.path.isfile(filename):
            raise FileExistsError
        return torch.load(filename, map_location=lambda s, loc: s)

    def load(self, epoch):
        """"
        Load the existing net parameters

        Parameters
        ----------
        epoch: str or int
            The epoch which specify the model

        Returns
        -------
        The initialized net
        """
        # 根据起始轮次装载已有的网络参数
        filename = self.epoch_params_filename(epoch)
        return self.load_net(
            filename
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

    @staticmethod
    def net_initialize(net, model_ctx="cpu", **kwargs):
        """初始化网络参数"""
        return set_device(net, model_ctx, **kwargs)
