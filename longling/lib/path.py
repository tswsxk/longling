# coding: utf-8
# create by tongshiwei on 2019/7/2

__all__ = ["path_append", "file_exist"]

import os
from pathlib import PurePath


def path_append(path, addition=None, to_str=False):
    """

    Parameters
    ----------
    path: str or PurePath
    addition: str or PurePath
    to_str: bool
        Convert the new path to str
    Returns
    -------

    """
    path = PurePath(path)
    new_path = path / addition if addition else path
    if to_str:
        return str(new_path)
    return new_path


def file_exist(file_path):
    return os.path.isfile(file_path)
