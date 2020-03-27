# coding: utf-8
# 2020/1/2 @ tongshiwei

import pathlib
from tqdm import tqdm
from longling import as_out_io, as_io, PATH_IO_TYPE, PATH_TYPE
import csv
import json

__all__ = ["csv2json", "json2csv", "load_csv", "load_json", "loading"]


def csv2json(src, tar, delimiter=",", **kwargs):
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

    Returns
    -------

    """
    with as_out_io(tar) as wf:
        csv_writer = None
        for line in tqdm(load_json(src), "json2csv: %s --> %s" % (src, tar)):
            if csv_writer is None:
                csv_writer = csv.DictWriter(wf, line.keys(), delimiter=delimiter, **kwargs)
                csv_writer.writeheader()
            csv_writer.writerow(line)


def load_file(src: PATH_IO_TYPE):
    with as_io(src) as f:
        for line in f:
            yield line


def load_csv(src: PATH_IO_TYPE, delimiter=",", **kwargs):
    """read the dict from csv"""
    with as_io(src) as f:
        field_names = [i for i in csv.reader([f.readline()], delimiter=delimiter, **kwargs)][0]

        for line in csv.DictReader(f, field_names, delimiter=delimiter, **kwargs):
            yield line


def load_json(src: PATH_IO_TYPE):
    with as_io(src) as f:
        for line in f:
            yield json.loads(line)


def loading(src: (PATH_IO_TYPE, ...), src_type=None):
    if isinstance(src, PATH_TYPE):
        suffix = pathlib.PurePath(src).suffix[1:]
        if suffix == "csv" or src_type == "csv":
            return load_csv(src)
        elif suffix == "json" or src_type == "json":
            return load_json(src)
        else:
            return load_file(src)
    elif callable(src):
        return src()
    return src
