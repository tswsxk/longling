# coding: utf-8
# create by tongshiwei on 2017/12/6

from __future__ import absolute_import
from __future__ import unicode_literals

import json

from collections import OrderedDict

from longling.base import *
from longling.lib.stream import *
from ..dataIterator import OriginIterator, OriginBatchIterator
from ..adapter import Single2BatchAdapter


class FileIterator(OriginIterator):
    def __init__(self, filename, encoding='utf-8', **kwargs):
        self.filename = filename
        self.encoding = encoding
        self.kwargs = kwargs
        self.filesource = rf_open(filename, encoding=encoding, **kwargs)

    def reset(self):
        self.filesource.seek(0)

    def next(self):
        return self.filesource.readline()


class FileBatchIterator(Single2BatchAdapter, FileIterator):
    def __init__(self, filename, encoding='utf-8', batch_size=100000, **kwargs):
        file_iterator = FileIterator(filename, encoding, **kwargs)
        super(FileBatchIterator, self).__init__(file_iterator, batch_size=batch_size)


class JsonDictIterator(FileIterator):
    def __init__(self, filename, encoding='utf-8', key_dict=None, **kwargs):
        super(JsonDictIterator, self).__init__(filename, encoding)
        assert key_dict is not None
        if isinstance(key_dict, dict):
            self.key_dict = OrderedDict(zip(key_dict.values(), key_dict.keys()))
        elif isinstance(key_dict, (list, tuple, set)):
            self.key_dict = OrderedDict(zip(key_dict, key_dict))
        elif key_dict == "*":
            self.key_dict = "*"
        else:
            raise TypeError("key_dict must be dict, list, tuple, set or str'*' ")

    def next(self):
        line = unistr(self.filesource.readline())
        data = json.loads(line, encoding='utf-8')
        if self.key_dict == "*":
            return data
        return OrderedDict([(key_map, data[key]) for key, key_map in self.key_dict])


class JsonDictBatchIterator(JsonDictIterator, OriginBatchIterator):
    def __init__(self, filename, encoding='utf-8', batch_size=1000, key_dict=None, **kwargs):
        super(JsonDictBatchIterator, self).__init__(filename, encoding, key_dict)
        self.batch_size = batch_size
        self.buffer = self.init_buffer()
        self.batch_cnt = 0

    def init_buffer(self):
        return OrderedDict(zip(self.key_dict.values(), [[] for _ in range(len(self.key_dict.values()))])) \
            if self.key_dict != "*" \
            else OrderedDict()

    def next_batch(self):
        while True:
            try:
                data = self.next()
                for key, value in data.items():
                    if self.key_dict == "*" and key not in self.buffer:
                        self.buffer[key] = []
                    elif key not in self.buffer:
                        continue
                    self.buffer[key].append(value)
                self.batch_cnt += 1
                if self.batch_cnt == self.batch_size:
                    return_data = self.buffer
                    self.buffer = self.init_buffer()
                    self.batch_cnt = 0
                    return return_data
                elif self.batch_cnt > self.batch_size:
                    raise Exception("self.buff is too big")
            except StopIteration:
                if self.batch_cnt > 0:
                    return_data = self.buffer
                    self.buffer = self.init_buffer()
                    self.batch_cnt = 0
                    return return_data
                raise StopIteration


class JsonxzIterator(FileIterator):
    def next(self):
        line = unistr(self.filesource.readline())
        data = json.loads(line, encoding='utf8')
        return data['x'], data['z']


class JsonxzBatchIterator(JsonxzIterator, OriginBatchIterator):
    def __init__(self, filename, encoding='utf-8', batch_size=1000, **kwargs):
        super(JsonxzBatchIterator, self).__init__(filename, encoding=encoding, **kwargs)
        self.batch_size = batch_size
        self.xs = []
        self.zs = []

    def next_batch(self):
        while True:
            try:
                x, z = self.next()
                self.xs.append(x)
                self.zs.append(z)
                if len(self.xs) == self.batch_size:
                    assert len(self.xs) == len(self.zs)
                    xs = self.xs
                    zs = self.zs
                    self.xs = []
                    self.zs = []
                    assert self.xs
                    assert self.zs
                    return xs, zs
                elif len(self.xs) > self.batch_size:
                    raise Exception("self.buff is too big")
            except StopIteration:
                if self.xs:
                    assert len(self.xs) == len(self.zs)
                    xs = self.xs
                    zs = self.zs
                    self.xs = []
                    self.zs = []
                    return xs, zs
                raise StopIteration


class CSVIterator(FileIterator):
    def __init__(self, filename, encoding='utf-8',
                 reserved_fields_name=None, reserved_fields_index=None, **kwargs):
        super(CSVIterator, self).__init__(filename, encoding, **kwargs)
        self.separator = kwargs.get('separator', ',')
        self.fields_name = self._get_fields_name()
        if reserved_fields_index is not None:
            self.reserved_fields_index = reserved_fields_index
        else:
            self.reserved_fields_index = self._get_reserved_fields_index(reserved_fields_name)
        self.reserved_name_index_pairs = self._get_reserved_name_index_pairs()

    def _get_reserved_name_index_pairs(self):
        reserved_name_index_pairs = []
        for i, name in enumerate(self.fields_name):
            if i in self.reserved_fields_index:
                reserved_name_index_pairs.append((name, i))
        return reserved_name_index_pairs

    def _get_reserved_fields_index(self, reserved_fields_name):
        if reserved_fields_name is None:
            return None
        reserved_fields_name = set(reserved_fields_name)
        reserved_fields_index = set()
        index_name_pairs = zip(self.fields_name, range(len(self.fields_name)))
        for index_name in index_name_pairs:
            index, name = index_name
            if name in reserved_fields_name:
                reserved_fields_index.add(index)
        return reserved_fields_index

    def _get_fields_name(self):
        fields_name = unistr(self.filesource.readline())
        fields_name = fields_name.strip().split(self.separator)
        self.fields_name = fields_name
        return self.fields_name

    def reset(self):
        self.filesource.seek(0)
        self.filesource.readline()

    def filter(self, data):
        if self.reserved_fields_index is None:
            return data
        return_data = []
        for i, d in enumerate(data):
            if i in self.reserved_fields_index:
                return_data.append(d)
        return return_data

    def next(self):
        return self.filter(unistr(self.filesource.readline()).strip().split(self.separator))


class CSVBatchIterator(Single2BatchAdapter, CSVIterator):
    def __init__(self, filename, encoding='utf-8',
                 reserved_fields_name=None, reserved_fields_index=None,
                 batch_size=100000, **kwargs):
        csv_iterator = CSVIterator(filename, encoding, reserved_fields_name, reserved_fields_index, **kwargs)
        super(CSVBatchIterator, self).__init__(csv_iterator, batch_size=batch_size)
