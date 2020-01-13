# coding: utf-8
# 2020/1/2 @ tongshiwei

import pathlib
from tqdm import tqdm
from longling import wf_open, rf_open, PATH_TYPE
import csv
import json

__all__ = ["csv2json", "json2csv", "load_csv", "load_json", "loading"]


def csv2json(src, tar, delimiter=",", **kwargs):
    with wf_open(tar) as wf:
        for line in tqdm(load_csv(src, delimiter=delimiter, **kwargs), "csv2json: %s --> %s" % (src, tar)):
            print(json.dumps(line), file=wf)


def json2csv(src, tar, delimiter=",", **kwargs):
    with wf_open(tar) as wf:
        csv_writer = None
        for line in tqdm(load_json(src), "json2csv: %s --> %s" % (src, tar)):
            if csv_writer is None:
                csv_writer = csv.DictWriter(wf, line.keys(), delimiter=delimiter, **kwargs)
                csv_writer.writeheader()
            csv_writer.writerow(line)


def load_file(src: PATH_TYPE):
    with rf_open(src) as f:
        for line in f:
            yield line


def load_csv(src: PATH_TYPE, delimiter=",", **kwargs):
    """read the dict from csv"""
    with rf_open(src) as f:
        field_names = [i for i in csv.reader([f.readline()], delimiter=delimiter, **kwargs)][0]

        for line in csv.DictReader(f, field_names, delimiter=delimiter, **kwargs):
            yield line


def load_json(src: PATH_TYPE):
    with rf_open(src) as f:
        for line in f:
            yield json.loads(line)


def loading(src: (PATH_TYPE, ...), src_type=None):
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
