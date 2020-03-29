# coding: utf-8
# 2020/1/2 @ tongshiwei

import logging
import pathlib
from tqdm import tqdm
from longling import as_out_io, as_io, PATH_IO_TYPE, PATH_TYPE, IO_TYPE
import csv
import json

logger = logging.getLogger("longling")

__all__ = ["csv2json", "json2csv", "loading", "load_jsonl", "load_csv", "load_file"]


def csv2json(src, tar, delimiter=",", **kwargs):
    """
    将 csv 格式文件/io流 转换为 json 格式文件/io流

    transfer csv file or io stream into  json file or io stream

    Parameters
    ----------
    src: PATH_IO_TYPE
        数据源，可以是文件路径，也可以是一个IO流。
        the path to source file or io stream.
    tar: PATH_IO_TYPE
        输出目标，可以是文件路径，也可以是一个IO流。
        the path to target file or io stream.
    delimiter: str
        分隔符
        the delimiter used in csv. some usually used delimiters are ","  and " "
    kwargs: dict
        options passed to csv.DictWriter
    """
    with as_out_io(tar) as wf:
        for line in tqdm(load_csv(src, delimiter=delimiter, **kwargs), "csv2json: %s --> %s" % (src, tar)):
            print(json.dumps(line), file=wf)


def json2csv(src: PATH_IO_TYPE, tar: PATH_IO_TYPE, delimiter=",", **kwargs):
    """
    将 json 格式文件/io流 转换为 csv 格式文件/io流

    transfer json file or io stream into csv file or io stream

    Parameters
    ----------
    src: PATH_IO_TYPE
        数据源，可以是文件路径，也可以是一个IO流。
        the path to source file or io stream.
    tar: PATH_IO_TYPE
        输出目标，可以是文件路径，也可以是一个IO流。
        the path to target file or io stream.
    delimiter: str
        分隔符
        the delimiter used in csv. some usually used delimiters are ","  and " "
    kwargs: dict
        options passed to csv.DictWriter
    """
    with as_out_io(tar) as wf:
        csv_writer = None
        for line in tqdm(load_jsonl(src), "json2csv: %s --> %s" % (src, tar)):
            if csv_writer is None:
                csv_writer = csv.DictWriter(wf, line.keys(), delimiter=delimiter, **kwargs)
                csv_writer.writeheader()
            csv_writer.writerow(line)


def load_file(src: PATH_IO_TYPE):
    """
    Read raw text from source

    Examples
    --------
    Assume such component is written in demo.txt:

    .. code-block::

        hello
        world

    use following codes to reading the component

    .. code-block:: python

        for line in load_csv('demo.txt'):
            print(line, end="")

    and get

    .. code-block::

        hello
        world
    """
    with as_io(src) as f:
        for line in f:
            yield line


def load_csv(src: PATH_IO_TYPE, delimiter=",", **kwargs):
    """
    read the dict from csv

    Examples
    --------

    Assume such component is written in demo.csv:

    .. code-block::

        a,b,c
        1,2,3
        2,4,6


    .. code-block:: python

        for line in load_csv('demo.csv'):
            print(line)


    .. code-block::

        {"a": 1, "b": 2, "c": 3}
        {"a": 2, "b": 4, "c": 6}
    """
    with as_io(src) as f:
        field_names = [i for i in csv.reader([f.readline()], delimiter=delimiter, **kwargs)][0]

        for line in csv.DictReader(f, field_names, delimiter=delimiter, **kwargs):
            yield line


def load_jsonl(src: PATH_IO_TYPE):
    """
    缓冲式按行读取jsonl文件

    Examples
    --------

    Assume such component is written in demo.jsonl:

    .. code-block::

        {"a": 1}
        {"a": 2}


    .. code-block:: python

        for line in load_jsonl('demo.jsonl'):
            print(line)


    .. code-block::

        {"a": 1}
        {"a": 2}
    """
    with as_io(src) as f:
        for line in f:
            yield json.loads(line)


def loading(src: (PATH_IO_TYPE, ...), src_type=None):
    """
    缓冲式按行读取文件

    Support read from

    * jsonl (apply load_jsonl)
    * csv (apply load_csv).
    * Other format will be treated as raw text (apply load_file).
    """
    if isinstance(src, PATH_TYPE):
        suffix = pathlib.PurePath(src).suffix[1:]
        if suffix == "csv" or src_type == "csv":
            return load_csv(src)
        elif suffix in {"json", "jsonl"} or src_type in {"json", "jsonl"}:
            if suffix == "json" or src_type == "json":
                logger.warning("detect source type as json, processed as jsonl format")
            return load_jsonl(src)
        else:
            return load_file(src)
    elif isinstance(src, IO_TYPE):
        return load_file(src)
    elif callable(src):
        return src()
    return src
