# coding: utf-8
# create by tongshiwei on 2019/7/2
# PYTEST_DONT_REWRITE

__all__ = ["path_append", "file_exist", "abs_current_dir"]

import os
from pathlib import PurePath


def path_append(path, *addition, to_str=False):
    """
    路径合并函数

    Examples
    --------
    >>> path_append("../", "../data", "../dataset1/", "train", to_str=True)
    '../../data/../dataset1/train'

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
    """判断文件是否存在"""
    return os.path.isfile(file_path)


def abs_current_dir(file_path):
    """获取文件所在目录的绝对路径"""
    return os.path.abspath(os.path.dirname(file_path))


if __name__ == '__main__':
    import doctest

    doctest.testmod()
