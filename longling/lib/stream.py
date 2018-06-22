# coding: utf-8

# 此模块用以进行流处理
from __future__ import print_function

import sys
import codecs
import os

from longling.base import string_types, type_assert


class StreamError(Exception):
    pass


def flush_print(*values, **kwargs):
    sep = kwargs.get('sep', "")
    end = kwargs.get('end', "")
    print('\r', *values, sep=sep, end=end, flush=True)


def _build_dir(path, mode=0o775):
    '''
    创建目录，从path中解析出目录路径，如果目录不存在，创建目录
    :param path:
    :param mode:
    :return:
    '''
    dirname = os.path.dirname(path)
    if not dirname or os.path.exists(dirname):
        return
    os.makedirs(dirname, mode)


def _check_file(path, size=None):
    '''
    检查文件是否存在，size给定时，检查文件大小是否一致
    :param path:
    :param size:
    :return:
    '''
    if os.path.exists(path):
        return size == os.path.getsize(path) if size is not None else True
    return False


def rf_open(filename, encoding='utf-8', **kwargs):
    if sys.version_info[0] == 3:
        return open(filename, encoding=encoding, **kwargs)
    else:
        return open(filename, **kwargs)


def _wf_open(stream_name='', mode="w", encoding="utf-8"):
    '''
    打开一个codecs流
    :param stream_name: str，默认为空
    :param mode: str，默认为 写 模式
    :param encoding: str，编码方式，默认为 utf-8
    :return: 返回打开的流，stream_name为空时 mode - w 返回标准错误输出; 否则，返回标准输出，不为空时，返回文件流
    '''
    if not stream_name:
        if mode == "w":
            return sys.stderr
        else:
            return sys.stdout
    elif isinstance(stream_name, string_types):
        build_dir(stream_name)
        if mode == "wb":
            return open(stream_name, mode=mode)
        return codecs.open(stream_name, mode=mode, encoding=encoding)
    else:
        raise StreamError("wf_open: %s" % stream_name)


def wf_close(stream):
    if stream in {sys.stdin, sys.stdout, sys.stderr}:
        return True
    else:
        try:
            stream.close()
        except Exception:
            raise StreamError('wf_close: %s' % stream)


# type checker
if sys.version_info[0] == 3:
    def wf_open(stream_name: string_types = '', mode: string_types = "w", encoding: string_types = "utf-8"):
        return _wf_open(stream_name, mode, encoding)


    def check_file(path: string_types, size: int = None) -> bool:
        return _check_file(path, size)


    def build_dir(path: string_types, mode: int = 0o775):
        return _build_dir(path, mode)
else:
    wf_open = type_assert(stream_name=string_types, mode=string_types, encoding=string_types)(_wf_open)
    check_file = type_assert(path=string_types, size=int)(_check_file)
    build_dir = type_assert(path=string_types, mode=int)(_build_dir)
