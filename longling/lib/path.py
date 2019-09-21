# coding: utf-8
# create by tongshiwei on 2019/7/2

__all__ = ["path_append", "file_exist", "abs_current_dir"]

import os
from pathlib import PurePath


def path_append(path, *addition, to_str=False):
    """

    Parameters
    ----------
    path: str or PurePath
    addition: list(str or PurePath)
    to_str: bool
        Convert the new path to str
    Returns
    -------

    """
    path = PurePath(path)
    if addition:
        for a in addition:
            path = path / a
    if to_str:
        return str(path)
    return path


def file_exist(file_path):
    return os.path.isfile(file_path)


def abs_current_dir(file_path):
    return os.path.abspath(os.path.dirname(file_path))
