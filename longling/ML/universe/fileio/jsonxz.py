# coding: utf-8
# created by tongshiwei on 17-12-9
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import json

from tqdm import tqdm

from ..dataIterator import CSVIterator

from longling.lib.stream import wf_open, wf_close


def load(source, data_key='x', label_key='z'):
    datas = []
    labels = []
    with open(source) as f:
        for line in tqdm(f):
            data = json.loads(line)
            x, z = data[data_key], data[label_key]
            datas.append(x)
            labels.append(z)
    return datas, labels


def dump(objs, fp, wrapper=tqdm):
    for obj in wrapper(objs):
        print(json.dumps(obj, ensure_ascii=False), file=fp)


def verify(filename):
    with open(filename) as f:
        for line in tqdm(f):
            json.loads(line)


def csv2jsonxz(source, target, label_index=None, label_name=None,
               reserved_fields_index=None, reserved_fields_name=None,
               **kwargs):

    if label_index is None and label_name is None:
        raise ValueError("label_index and label_name both are None")

    if reserved_fields_name is not None:
        reserved_fields_name = set(reserved_fields_name + [label_name])
    if reserved_fields_index is not None:
        reserved_fields_index = set(reserved_fields_index + [label_index])

    datas = CSVIterator(source,
                        reserved_fields_name=reserved_fields_name, reserved_fields_index=reserved_fields_index,
                        *kwargs)
    if label_index is None:
        label_index = dict(datas.reserved_name_index_pairs)[label_name]

    data_key = kwargs.get('data_key', 'x')
    label_key = kwargs.get('label_key', 'z')

    wf = wf_open(target)
    for data in datas:
        x, z = [], None
        for i, d in enumerate(data):
            if i == label_index:
                z = d
            else:
                x.append(d)
        xz = {data_key: x, label_key: z}
        print(json.dumps(xz, ensure_ascii=False), file=wf)
    wf_close(wf)
