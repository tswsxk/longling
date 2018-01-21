# coding: utf-8

# 此模块用以进行流处理

import sys
import codecs
import os

from longling.base import string_types

def checkDir(path, mode=0o777):
    dirname = os.path.dirname(path)
    if not dirname or os.path.exists(dirname):
        return
    os.makedirs(dirname, mode)


def rf_open(filename, encoding='utf-8', **kwargs):
    if sys.version_info[0] == 3:
        return open(filename, encoding=encoding, **kwargs)
    else:
        return open(filename, **kwargs)


def wf_open(stream_name='', mode="w", encoding="utf-8"):
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
        checkDir(stream_name)
        return codecs.open(stream_name, mode=mode, encoding=encoding)
    else:
        raise TypeError("wf_open: %s" % stream_name)


def wf_close(stream):
    if stream in {sys.stdin, sys.stdout, sys.stderr}:
        return True
    else:
        try:
            stream.close()
        except:
            raise Exception('wf_close: %s' % stream)
