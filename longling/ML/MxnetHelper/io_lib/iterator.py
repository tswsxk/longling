# coding: utf-8
# created by tongshiwei on 18-1-27

from __future__ import absolute_import

import bisect
import json
import logging
import random

from collections import OrderedDict, namedtuple, Iterable

import mxnet as mx
import numpy as np

from tqdm import tqdm

Desc = namedtuple('Desc', ['shape', 'dtype'])


def extract_list(lists):
    return lists


def get_batch_size(iter):
    return iter.provide_data[0].shape[0]


class MulIter(mx.io.DataIter):
    def __init__(self, batch_size, iter_list):
        super(MulIter, self).__init__(batch_size)
        self.iters = iter_list

    def next(self):
        batches = [i.next() for i in self.iters]
        return mx.io.DataBatch(data=[extract_list(*b.data) for b in batches],
                               label=[extract_list(*b.label) for b in batches if b.label])

    def reset(self):
        for i in self.iters:
            i.reset()

    @property
    def provide_data(self):
        return [extract_list(*i.provide_data) for i in self.iters]

    @property
    def provide_label(self):
        return [extract_list(*i.provide_label) for i in self.iters if i.provide_label]


class DictJsonIter(mx.io.DataIter):
    def __init__(self, batch_size, filename, data_key_dict={'data': 'x'}, label_key_dict={'label': 'z'},
                 last_batch_handle='pad'):
        super(DictJsonIter, self).__init__(batch_size)
        self.f = open(filename)
        self.data_key_dict = OrderedDict(data_key_dict)
        self.label_key_dict = OrderedDict(label_key_dict)
        self.data_desc = {}
        self.label_desc = {}
        self.data = [[] for _ in range(len(self.data_key_dict))]
        self.label = [[] for _ in range(len(self.label_key_dict))]
        self.cnt = 0
        self.index = 0
        self.batch_size = batch_size
        self.last_batch_handle = last_batch_handle
        self.end_tag = False
        self.determine_data_shapes()
        self.pad = 0

    def determine_data_shapes(self):
        line = self.f.readline()
        datas = json.loads(line)
        for i, name in enumerate(self.data_key_dict):
            data = np.asarray(datas[self.data_key_dict[name]], dtype=np.float32)
            self.data_desc[name] = Desc(data.shape, data.dtype)
        for i, name in enumerate(self.label_key_dict):
            data = np.asarray(datas[self.label_key_dict[name]], dtype=np.float32)
            self.label_desc[name] = Desc(data.shape, data.dtype)
        self.reset()

    def iter_next(self):
        try:
            if self.end_tag:
                raise StopIteration
            self.data = [[] for _ in range(len(self.data_key_dict))]
            self.label = [[] for _ in range(len(self.label_key_dict))]
            for line in self.f:
                if not line.strip():
                    continue
                datas = json.loads(line)
                for i, name in enumerate(self.data_key_dict):
                    self.data[i].append(np.asarray(datas[self.data_key_dict[name]], dtype=np.float32))
                for i, name in enumerate(self.label_key_dict):
                    self.label[i].append(np.asarray(datas[self.label_key_dict[name]], dtype=np.float32))
                self.cnt += 1
                if self.cnt >= self.batch_size:
                    self.cnt = 0
                    self.index += 1
                    return True
            raise StopIteration
        except StopIteration:
            if self.last_batch_handle == "discard":
                self.data = [[] for _ in range(len(self.data_key_dict))]
                self.label = [[] for _ in range(len(self.label_key_dict))]
                return False
            elif not self.end_tag and self.last_batch_handle in {"roll_over", "pad"}:
                self.pad = self.batch_size - self.cnt
                self.f.seek(0)
                for i, line in enumerate(self.f):
                    if i >= self.pad:
                        break
                    datas = json.loads(line)
                    for i, name in enumerate(self.data_key_dict):
                        self.data[i].append(np.asarray(datas[self.data_key_dict[name]], dtype=np.float32))
                    for i, name in enumerate(self.label_key_dict):
                        self.label[i].append(np.asarray(datas[self.label_key_dict[name]], dtype=np.float32))
                self.index += 1
                self.end_tag = True
                return True
            else:
                self.data = [[] for _ in range(len(self.data_key_dict))]
                self.label = [[] for _ in range(len(self.label_key_dict))]
                return False

    def getdata(self):
        return [mx.nd.array(np.asarray(d)) for d in self.data]

    def getlabel(self):
        return [mx.nd.array(np.asarray(l)) for l in self.label]

    def getindex(self):
        return self.index

    def reset(self):
        self.index = 0
        self.cnt = 0
        if self.last_batch_handle == "roll_over" and self.end_tag:
            pass
        else:
            self.f.seek(0)
        self.end_tag = False
        self.pad = 0
        self.data = [[] for _ in range(len(self.data_key_dict))]
        self.label = [[] for _ in range(len(self.label_key_dict))]

    def getpad(self):
        return self.pad

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator."""
        return [
            mx.io.DataDesc(k, tuple([self.batch_size] + list(v.shape)), v.dtype)
            for k, v in self.data_desc.items()
        ]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator."""
        return [
            mx.io.DataDesc(k, tuple([self.batch_size] + list(v.shape)), v.dtype)
            for k, v in self.label_desc.items()
        ]


class SimpleBucketIter(mx.io.DataIter):
    def __init__(self, batch_size, data, label=None, data_name='data', label_name='label',
                 buckets=[], dtype='float32', padding_num=0, layout='NT', label_shape=1,
                 for_predicting=False, shuffle=True,
                 logger=logging):
        '''
        last batch handles for each batch are all discard in training
        and pad for predicting
        :param batch_size:
        :param data:
        :param label:
        :param data_name:
        :param label_name:
        :param buckets:
        :param dtype:
        :param padding_num:
        :param layout:
        :param label_shape:
        :param logger:
        '''
        super(SimpleBucketIter, self).__init__(batch_size)

        if not buckets:
            buckets = [i for i, j in enumerate(np.bincount([len(d) for d in data]))
                       if j >= self.batch_size]
        buckets.sort()

        ndiscard = 0
        self.data = [[] for _ in buckets]
        self.label = [[] for _ in buckets] if label is not None else None
        self.pad = [0 for _ in buckets]
        self.label_shape = label_shape

        self.bucket_distribution(
            data=data,
            data_buffs=[self.data],
            buckets=buckets,
            label=label,
            label_buffs=[self.label],
            padding_num=padding_num,
            dtype=dtype,
        )

        if for_predicting:
            for i, d in enumerate(self.data):
                if not d:
                    continue
                pad = self.get_padding_num(d, self.batch_size)
                if pad != self.batch_size:
                    self.pad[i] = pad
                    self._padding(d, [padding_num] * buckets[i], pad)

        self.data = [mx.ndarray.array(np.asarray(d, dtype=dtype)) for d in self.data]
        self.label = [mx.ndarray.array(np.asarray(l)) for l in self.label] if self.label is not None else None

        logger.info("%s: discarded %d sentences longer than the largest bucket." % (self.__class__, ndiscard))

        self.batch_size = batch_size
        self.buckets = buckets
        self.data_name = data_name
        self.label_name = label_name
        self.dtype = dtype
        self.invalid_label = padding_num
        self.major_axis = layout.find('N')
        self.layout = layout

        self.default_bucket_key = max(buckets)

        self.idx = []
        for i, buck in enumerate(self.data):
            self.idx.extend([(i, j) for j in range(0, len(buck) - batch_size + 1, batch_size)])
        self.curr_idx = 0

        self.for_predicting = for_predicting
        self.shuffle = shuffle
        self.reset()

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        self.curr_idx = 0
        if self.shuffle and not self.for_predicting:
            random.shuffle(self.idx)

    def next(self):
        if self.curr_idx == len(self.idx):
            raise StopIteration
        i, j = self.idx[self.curr_idx]
        self.curr_idx += 1

        if self.major_axis == 1:
            data = self.data[i][j:j + self.batch_size].T
            label = self.label[i][j:j + self.batch_size].T if self.label is not None else None
        else:
            data = self.data[i][j:j + self.batch_size]
            label = self.label[i][j:j + self.batch_size] if self.label is not None else None

        return mx.io.DataBatch([data], [label] if self.label is not None else None, pad=0,
                               bucket_key=self.buckets[i],
                               provide_data=[mx.io.DataDesc(
                                   name=self.data_name, shape=data.shape,
                                   layout=self.layout)],
                               provide_label=[mx.io.DataDesc(
                                   name=self.label_name, shape=label.shape,
                                   layout=self.layout)] if self.label is not None else self.provide_label)

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator."""
        if self.major_axis == 0:
            return [mx.io.DataDesc(
                name=self.data_name, shape=(self.batch_size, self.default_bucket_key),
                layout=self.layout)]
        elif self.major_axis == 1:
            return [mx.io.DataDesc(
                name=self.data_name, shape=(self.default_bucket_key, self.batch_size),
                layout=self.layout)]
        else:
            raise ValueError("Invalid layout %s: Must by NT (batch major) or TN (time major)")

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator."""
        if self.label_shape is not None:
            if self.label_shape == 1:
                shape = (self.batch_size,)
            elif isinstance(self.label_shape, str):
                shape = eval(self.label_shape)
            elif isinstance(self.label_shape, tuple):
                shape = (self.batch_size,) + self.label_shape
            else:
                raise ValueError()

            return [mx.io.DataDesc(name=self.label_name, shape=shape, layout=self.layout)]

        else:
            if self.major_axis == 0:
                return [mx.io.DataDesc(
                    name=self.label_name, shape=(self.batch_size, self.default_bucket_key),
                    layout=self.layout)]
            elif self.major_axis == 1:
                return [mx.io.DataDesc(
                    name=self.label_name, shape=(self.default_bucket_key, self.batch_size),
                    layout=self.layout)]
            else:
                raise ValueError("Invalid layout %s: Must by NT (batch major) or TN (time major)")

    @staticmethod
    def bucket_sort(iterator, sorted_key=lambda x: len(x)):
        if isinstance(iterator, Iterable):
            pass
        else:
            raise TypeError("SimpleBucketIter: iterator must be Iterable")

        buff = []

        for idx, data in tqdm(enumerate(iterator), desc="SimpleBucketIter loading data"):
            buff.append((data, sorted_key(data), idx))

        buff.sort(key=lambda x: x[1])

        return zip(*buff)

    @staticmethod
    def bucket_distribution(data, data_buffs, buckets, label=None, label_buffs=None, padding_num=0, dtype='float32',
                            cut_off=False):
        ndiscard = 0
        for i, d in enumerate(data):
            buck = bisect.bisect_left(buckets, len(d))
            if buck == len(buckets):
                if not cut_off:
                    ndiscard += 1
                    continue
                else:
                    buck -= 1
                    d = d[:buckets[buck]]
            buff = np.full((buckets[buck],), padding_num, dtype=dtype)
            buff[:len(d)] = d
            for data_buff in data_buffs:
                if data_buff is not None:
                    data_buff[buck].append(buff)
            if label is not None and label_buffs is not None:
                for label_buff in label_buffs:
                    if label_buff is not None:
                        label_buff[buck].append(label[i])
        return ndiscard

    @staticmethod
    def get_padding_num(data, batch_size):
        return batch_size - len(data) % batch_size

    @staticmethod
    def _padding(data, padding, padding_times):
        data.extend(np.asarray([padding] * padding_times))

    @staticmethod
    def padding(data, batch_size, padding):
        padding_times = SimpleBucketIter.get_padding_num(data, batch_size)
        SimpleBucketIter._padding(data, padding, padding_times)
        return padding_times


class BucketIter(mx.io.DataIter):
    pass
