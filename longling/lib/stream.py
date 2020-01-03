# coding: utf-8

"""此模块用以进行流处理"""
# todo 完善注释

from __future__ import print_function

from tqdm import tqdm
import codecs
import os
import sys
from pathlib import PurePath

__all__ = ['rf_open', 'wf_open', 'wf_close', 'flush_print', 'build_dir', 'json_load',
           'pickle_load', 'AddPrinter', 'AddObject', 'StreamError', 'check_file', 'PATH_TYPE', 'encoding']


class StreamError(Exception):
    pass


PATH_TYPE = (str, PurePath)


def flush_print(*values, **kwargs):
    sep = kwargs.get('sep', "")
    end = kwargs.get('end', "")
    print('\r', *values, sep=sep, end=end, flush=True)


def build_dir(path, mode=0o775):
    """
    创建目录，从path中解析出目录路径，如果目录不存在，创建目录

    Parameters
    ----------
    path: str
    mode: int

    """
    dirname = os.path.dirname(path)
    if not dirname or os.path.exists(dirname):
        return path
    os.makedirs(dirname, mode)
    return path


def check_file(path, size=None):
    """
    检查文件是否存在，size给定时，检查文件大小是否一致

    Parameters
    ----------
    path: str
    size: int

    Returns
    -------
    file exist or not: bool

    """
    if os.path.exists(path):
        return size == os.path.getsize(path) if size is not None else True
    return False


def rf_open(filename, encoding='utf-8', **kwargs):
    if kwargs.get("mode"):
        return open(filename, **kwargs)
    return open(filename, encoding=encoding, **kwargs)


def json_load(fp, **kwargs):
    import json
    if isinstance(fp, PATH_TYPE):
        fp = rf_open(fp, **kwargs)
        datas = json.load(fp, **kwargs)
        fp.close()
        return datas
    else:
        return json.load(fp, **kwargs)


def pickle_load(fp, encoding='utf-8', mode='rb', **kwargs):
    import pickle
    if isinstance(fp, PATH_TYPE):
        fp = open(fp, mode=mode, **kwargs)
        datas = pickle.load(fp, encoding=encoding, **kwargs)
        fp.close()
        return datas
    else:
        return pickle.load(fp, encoding=encoding, **kwargs)


def wf_open(stream_name: (PATH_TYPE, None) = '', mode="w", encoding="utf-8"):
    r"""
    Simple wrapper to codecs for writing.

    stream_name为空时 mode - w 返回标准错误输出 stderr;
    否则，返回标准输出 stdout

    stream_name不为空时，返回文件流

    Parameters
    ----------
    stream_name: str, PurePath or None
    mode: str
    encoding: str
        编码方式，默认为 utf-8
    Returns
    -------
    write_stream: StreamReaderWriter
        返回打开的流

    Examples
    --------
    >>> wf = wf_open(mode="stdout")
    >>> print("hello world", file=wf)
    hello world
    """
    if not stream_name:
        if mode == "stderr":
            return sys.stderr
        elif mode == "stdout":
            return sys.stdout
        else:
            raise TypeError("Unknown mode for std mode, only `stdout` and `stderr` are supported.")
    elif isinstance(stream_name, PATH_TYPE):
        build_dir(stream_name)
        if mode == "wb":
            return open(stream_name, mode=mode)
        return codecs.open(stream_name, mode=mode, encoding=encoding)
    else:
        raise TypeError("unknown type: %s(%s)" % (stream_name, type(stream_name)))


def wf_close(stream):
    """关闭文件流，忽略 sys.stdin, sys.stdout, sys.stderr"""
    if stream in {sys.stdin, sys.stdout, sys.stderr}:
        return True
    else:
        try:
            stream.close()
        except Exception:
            raise StreamError('wf_close: %s' % stream)


def encoding(src, src_encoding, tar, tar_encoding):
    with rf_open(src, encoding=src_encoding) as f, wf_open(tar, encoding=tar_encoding) as wf:
        for line in tqdm(f, "encoding %s[%s] --> %s[%s]" % (src, src_encoding, tar, tar_encoding)):
            print(line, end='', file=wf)


class AddObject(object):
    def add(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


class AddPrinter(AddObject):
    """
    以add方法添加文件内容的打印器

    Examples
    --------
    >>> import sys
    >>> printer = AddPrinter(sys.stdout)
    >>> printer.add("hello world")
    hello world
    """

    def __init__(self, fp, values_wrapper=lambda x: x, **kwargs):
        super(AddPrinter, self).__init__()
        self.fp = fp
        self.value_wrapper = values_wrapper
        self.kwargs = kwargs

    def add(self, value):
        print(self.value_wrapper(value), file=self.fp, **self.kwargs)
