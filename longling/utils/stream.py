# coding: utf-8

"""此模块用以进行流处理"""
# todo 完善注释

from __future__ import print_function

import json
from _io import TextIOWrapper
from contextlib import contextmanager
from tqdm import tqdm
import codecs
import os
import sys
from pathlib import PurePath
import fileinput
from typing import BinaryIO, TextIO

__all__ = ['to_io', 'as_io', 'as_out_io', 'rf_open', 'wf_open', 'rf_group_open', 'wf_group_open', 'close_io',
           'flush_print', 'build_dir', 'json_load',
           'as_io_group', 'as_out_io_group',
           'pickle_load', 'AddPrinter', 'AddObject', 'StreamError', 'check_file',
           'PATH_TYPE', 'encode', 'IO_TYPE', "PATH_IO_TYPE", "block_std"
           ]


class StreamError(Exception):
    pass


PATH_TYPE = (str, PurePath)
IO_TYPE = (TextIOWrapper, TextIO, BinaryIO, codecs.StreamReaderWriter, fileinput.FileInput)
PATH_IO_TYPE = (PATH_TYPE, IO_TYPE)


@contextmanager
def block_std():
    """
    Examples
    --------
    >>> print("hello world")
    hello world
    >>> with block_std():
    ...     print("hello world")
    """
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    yield
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


def flush_print(*values, **kwargs):
    """刷新打印函数"""
    sep = kwargs.pop('sep', "")
    end = kwargs.pop('end', "")
    print('\r', *values, sep=sep, end=end, flush=True, **kwargs)


def build_dir(path, mode=0o775, parse_dir=True):
    """
    创建目录，从path中解析出目录路径，如果目录不存在，创建目录

    Parameters
    ----------
    path: str
    mode: int
    parse_dir: bool

    """
    if parse_dir:
        dirname = os.path.dirname(path)
    else:
        dirname = path
    if not dirname or os.path.exists(dirname):
        return path
    os.makedirs(dirname, mode)
    return path


def check_file(filepath, size=None):
    """
    检查文件是否存在，size给定时，检查文件大小是否一致

    Parameters
    ----------
    filepath: str
    size: int

    Returns
    -------
    file exist or not: bool

    """
    if os.path.exists(filepath):
        return size == os.path.getsize(filepath) if size is not None else True
    return False


def rf_open(stream_name: (PATH_IO_TYPE, list, None) = None, encoding='utf-8', **kwargs):
    if stream_name is None:
        return sys.stdin
    if isinstance(stream_name, list):
        if encoding is None:
            return fileinput.input(files=stream_name, **kwargs)
        else:
            return fileinput.input(
                files=stream_name,
                openhook=fileinput.hook_encoded(encoding)
            )
    elif isinstance(stream_name, IO_TYPE):
        return stream_name
    elif isinstance(stream_name, PATH_TYPE):
        encoding = None if kwargs.get("mode") == "rb" else encoding
        return open(stream_name, encoding=encoding, **kwargs)
    else:
        raise TypeError("unknown type: %s(%s)" % (stream_name, type(stream_name)))


def rf_group_open(*files, encoding="utf-8", **kwargs):
    return [rf_open(_file, encoding=encoding, **kwargs) for _file in files]


def json_load(fp: (IO_TYPE, PurePath), **kwargs):
    with as_io(fp) as f:
        return json.load(f, **kwargs)


def pickle_load(fp, encoding='utf-8', mode='rb', **kwargs):
    import pickle
    if isinstance(fp, PATH_TYPE):
        fp = open(fp, mode=mode, **kwargs)
        datas = pickle.load(fp, encoding=encoding, **kwargs)
        fp.close()
        return datas
    else:
        return pickle.load(fp, encoding=encoding, **kwargs)


def wf_open(stream_name: (PATH_IO_TYPE, None) = None, mode="w", encoding="utf-8", **kwargs):
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
    if stream_name is None:
        if stream_name is None and mode in {"w", "wb", "a", "ab"}:
            return sys.stdout
        elif mode == "stdout":
            return sys.stdout
        elif mode == "stderr":
            return sys.stderr
        else:
            raise TypeError("Unknown mode for std mode, only `stdout` and `stderr` are supported.")
    elif isinstance(stream_name, PATH_TYPE):
        build_dir(stream_name)
        if mode == "wb":
            return open(stream_name, mode=mode, **kwargs)
        return codecs.open(stream_name, mode=mode, encoding=encoding, **kwargs)
    elif isinstance(stream_name, IO_TYPE):
        return stream_name
    else:
        raise TypeError("unknown type: %s(%s)" % (stream_name, type(stream_name)))


def wf_group_open(*files, encoding="utf-8", **kwargs):
    return [wf_open(_file, encoding=encoding, **kwargs) for _file in files]


def to_io(stream: (TextIOWrapper, TextIO, BinaryIO, PATH_TYPE, list, None) = None, mode='r', encoding='utf-8',
          **kwargs):
    """
    Convert an object as an io stream, could be a path to file or an io stream.

    Examples
    --------

    .. code-block:: python

        to_io("demo.txt")  # equal to open("demo.txt")
        to_io(open("demo.txt"))  # equal to open("demo.txt")
        a = to_io()  # equal to a = sys.stdin
        b = to_io(mode="w)  # equal to a = sys.stdout
    """
    if isinstance(stream, IO_TYPE):
        return stream
    if 'r' in mode:
        return rf_open(stream, encoding=encoding, mode=mode, **kwargs)
    elif 'w' in mode or 'a' in mode or 'stderr' in mode or 'stdout' in mode:
        return wf_open(stream_name=stream, mode=mode, encoding=encoding)
    else:
        raise ValueError("cannot handle the mode %s" % mode)


def to_io_group(*streams, mode='r', encoding='utf-8', **kwargs):
    return [to_io(stream, mode, encoding, **kwargs) for stream in streams]


def close_io(stream):
    """关闭文件流，忽略 sys.stdin, sys.stdout, sys.stderr"""
    if isinstance(stream, list):
        for _stream in stream:
            close_io(_stream)
    elif stream in {sys.stdin, sys.stdout, sys.stderr}:
        pass
    else:
        try:
            stream.close()
        except Exception:
            raise StreamError('io_close: %s' % stream)


@contextmanager
def as_io(src: (TextIOWrapper, TextIO, BinaryIO, PATH_TYPE, list, None) = None, mode='r', encoding='utf-8', **kwargs):
    """
    with wrapper for to_io function, default mode is "r"

    Examples
    --------

    .. code-block:: python

        with as_io("demo.txt") as f:
            for line in f:
                pass

        # equal to
        with open(demo.txt) as src:
            with as_io(src) as f:
                for line in f:
                    pass

        # from several files
        with as_io(["demo1.txt", "demo2.txt"]) as f:
            for line in f:
                pass

        # from sys.stdin
        with as_io() as f:
            for line in f:
                pass
    """
    _io_object = to_io(src, mode, encoding, **kwargs)
    yield _io_object
    close_io(_io_object)


@contextmanager
def as_io_group(*src, mode='r', encoding='utf-8', **kwargs):
    ios = to_io_group(*src, mode=mode, encoding=encoding, **kwargs)
    yield ios
    close_io(ios)


@contextmanager
def as_out_io(tar: (TextIOWrapper, TextIO, BinaryIO, PATH_TYPE, list, None) = None, mode='w', encoding='utf-8',
              **kwargs):
    """
    with wrapper for to_io function, default mode is "w"

    Examples
    --------

    .. code-block:: python

        with as_out_io("demo.txt") as wf:
            print("hello world", file=wf)

        # equal to
        with open(demo.txt) as tar:
            with as_out_io(tar) as f:
                print("hello world", file=wf)

        # to sys.stdout
        with as_out_io() as wf:
            print("hello world", file=wf)

        # to sys.stderr
        with as_out_io(mode="stderr) as wf:
            print("hello world", file=wf)
    """
    with as_io(tar, mode, encoding, **kwargs) as out_io:
        yield out_io


@contextmanager
def as_out_io_group(*src, mode='w', encoding='utf-8', **kwargs):
    out_ios = to_io_group(*src, mode=mode, encoding=encoding, **kwargs)
    yield out_ios
    close_io(out_ios)


def encode(src, src_encoding, tar, tar_encoding):
    """
    Convert a file in source encoding to target encoding

    Parameters
    ----------
    src
    src_encoding
    tar
    tar_encoding

    Returns
    -------

    """
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
    >>> printer = AddPrinter(sys.stdout, ensure_io=True)
    >>> printer.add("hello world")
    hello world
    """

    def __init__(self, fp, values_wrapper=lambda x: x, to_io_params=None, ensure_io=False, **kwargs):
        super(AddPrinter, self).__init__()
        self.fp = wf_open(fp, **to_io_params if to_io_params is not None else {}) if not ensure_io else fp
        self.value_wrapper = values_wrapper
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def add(self, value):
        print(self.value_wrapper(value), file=self.fp, **self.kwargs)

    def close(self):
        close_io(self.fp)

    @property
    def name(self):
        return self.fp.name
