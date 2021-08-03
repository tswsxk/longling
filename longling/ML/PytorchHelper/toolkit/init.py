# coding: utf-8
# 2021/3/13 @ tongshiwei

import torch
import os
from .parallel import set_device


def net_initialize(net, model_ctx, *args, **kwargs):
    return set_device(net, model_ctx, *args, **kwargs)


def load_net(filename, net, allow_missing=False, *args, **kwargs):
    """
        Load the existing net parameters

        Parameters
        ----------
        filename: str
            The model file
        net: torch.nn.Module
            The network which has been initialized or
            loaded from the existed model
        allow_missing: bool

        Returns
        -------
        The initialized net
        """
    # 根据文件名装载已有的网络参数
    if not os.path.isfile(filename):
        raise FileExistsError("%s does not exist" % filename)
    net.load_state_dict(torch.load(filename), strict=not allow_missing)
    return net
