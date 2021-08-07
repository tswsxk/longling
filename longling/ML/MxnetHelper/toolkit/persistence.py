# coding: utf-8
# create by tongshiwei on 2020-11-12

import re

import mxnet.ndarray as nd

__all__ = ["save_params"]


def save_params(filename, net, select=None):
    """
    New in version 1.3.16

    Notes
    ------
    Q: Why not use the `save_parameters` in `mxnet.gluon`.
    A: Because we want to use the `select` argument to only preserve those parameters we want
    (i.e., excluding those unnecessary parameters such as pretrained embedding).
    """
    params = net._collect_params_with_prefix()
    if select is not None:
        pattern = re.compile(select)
        params = {name: value for name, value in params.items() if
                  pattern.match(name)}
    arg_dict = {key: val._reduce() for key, val in params.items()}
    nd.save(filename, arg_dict)
    return filename
