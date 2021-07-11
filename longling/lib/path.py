# coding: utf-8
# create by tongshiwei on 2019/7/2
# PYTEST_DONT_REWRITE

__all__ = ["path_append", "file_exist", "abs_current_dir", "type_from_name", "parent_dir", "tmpfile"]

import os
from pathlib import PurePath
import uuid
import tempfile
from contextlib import contextmanager


def type_from_name(filename):
    """
    Examples
    --------
    >>> type_from_name("1.txt")
    '.txt'
    """
    return PurePath(filename).suffix


def path_append(path, *addition, to_str=False):
    """
    路径合并函数

    Examples
    --------
    .. code-block:: python

    path_append("../", "../data", "../dataset1/", "train", to_str=True)
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


def file_exist(filepath):
    """判断文件是否存在"""
    return os.path.isfile(filepath)


def abs_current_dir(filepath):
    """
    获取文件所在目录的绝对路径

    Example
    -------
    .. code ::

        abs_current_dir(__file__)

    """
    return os.path.abspath(os.path.dirname(filepath))


def parent_dir(filepath, level):
    _path = filepath
    for _ in range(level):
        _path = path_append(_path, "..")

    return _path


@contextmanager
def tmpfile(suffix=None, prefix=None, dir=None):
    """
    Create a temporary file, which will automatically cleaned after used (outside "with" closure).

    Examples
    --------

    .. code-block ::

        with tmpfile("test_tmp") as tmp:
            print(tmp)
            with open(tmp, mode="w") as wf:
                print("hello world", file=wf)
    """
    prefix = prefix if prefix is not None else tempfile.gettempprefix() + str(uuid.uuid4())[:6]
    filename = prefix + suffix if suffix is not None else prefix
    with tempfile.TemporaryDirectory(dir=dir) as tmpdir:
        yield os.path.join(tmpdir, filename)
