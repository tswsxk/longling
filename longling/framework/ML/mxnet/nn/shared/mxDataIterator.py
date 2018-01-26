# coding: utf-8
# create by tongshiwei on 2017/10/23
from abc import ABCMeta, abstractmethod
import json
import logging
import os


import mxnet as mx
import numpy as np
from tqdm import tqdm

from collections import OrderedDict, namedtuple

Desc = namedtuple('Desc', ['shape', 'dtype'])


class originIterator():
    __metaclass__ = ABCMeta

    def tolist(self):
        listbuff = []
        for elem in self:
            listbuff.extend(elem)
        return listbuff

    def __iter__(self):
        return self

    def __next__(self, **kwargs):
        return self.next(**kwargs)

    @abstractmethod
    def next(self, **kwargs):
        pass

    @abstractmethod
    def next_batch(self, batch_size):
        pass

    def close(self):
        pass

    def reset(self):
        pass


def getNumIterator(location, channel=1, feature_num=None, vec_size=None, isString=False, data_key='x',
                   label_key='z', threshold=2, factor=1.0 / 8, logger=logging):
    filesize = float(os.path.getsize(location)) / (1024 * 1024 * 1024)
    filesize = round(filesize, 4)

    try:
        import psutil
        mem = psutil.virtual_memory()
        threshold = round(float(mem.free) / (1024 * 1024 * 1024) * factor, 6)
    except Exception as e:
        logger.warning(e)
        pass

    if filesize > threshold:
        iterator = NumIterator
        logger.info(
            "the size of file %s is %s GB, big data, threshold is %s GB, use NumIterator" % (
                location, filesize, threshold))
    else:
        iterator = numIterator
        logger.info(
            "the size of file %s is %s GB, small data, threshold is %s GB, use numIterator" % (
                location, filesize, threshold))

    return iterator(
        location=location,
        channel=channel,
        feature_num=feature_num,
        vec_size=vec_size,
        isString=isString,
        data_key=data_key,
        label_key=label_key
    )


class NumIterator(originIterator):
    def __init__(self, location, channel=1, feature_num=None, vec_size=None, isString=False, data_key='x',
                 label_key='z'):
        self.data_buffer = []
        self.label_buffer = []
        self.location = location
        self.channel = channel
        self.feature_num = feature_num
        self.vec_size = vec_size
        self.isString = isString
        self.data_key = data_key
        self.label_key = label_key

        self.cnt = 0
        with open(self.location) as fin:
            for _ in tqdm(fin):
                self.cnt += 1

        self.fin = open(self.location)

    def next_batch(self, batch_size):
        try:
            for line in self.fin:
                data = json.loads(line, encoding='utf8')
                datas = data[self.data_key]
                if self.isString:
                    datas = [float(w) for w in data[self.data_key].split()]
                if self.feature_num is not None and self.vec_size is not None:
                    self.data_buffer.append(
                        np.reshape(datas, newshape=(self.channel, self.feature_num, self.vec_size)))
                else:
                    self.data_buffer.append(datas)
                self.label_buffer.append(int(data[self.label_key]))

                assert len(self.label_buffer) == len(self.data_buffer)

                if len(self.data_buffer) == batch_size:
                    data_buffer = self.data_buffer
                    label_buffer = self.label_buffer
                    self.data_buffer = []
                    self.label_buffer = []
                    assert data_buffer and label_buffer
                    return np.array(data_buffer), np.array(label_buffer)
                elif len(self.data_buffer) > batch_size:
                    raise Exception("self.buff is too big")
            raise StopIteration
        except:
            assert len(self.label_buffer) == len(self.data_buffer)
            if self.data_buffer and self.label_buffer:
                data_buffer = self.data_buffer
                label_buffer = self.label_buffer
                self.data_buffer = []
                self.label_buffer = []
                return np.array(data_buffer), np.array(label_buffer)
            raise StopIteration

    def next(self, batch_size):
        return self.next_batch(batch_size)

    def reset(self):
        self.fin.seek(0)

    def close(self):
        self.fin.close()


class numIterator(originIterator):
    def __init__(self, location, channel=1, feature_num=None, vec_size=None, isString=False, data_key='x',
                 label_key='z'):
        self.cnt = 0
        self.data_buffer = []
        self.label_buffer = []
        with open(location) as fin:
            for line in tqdm(fin):
                data = json.loads(line, encoding='utf8')
                datas = data[data_key]
                if isString:
                    datas = [float(w) for w in data[data_key].split()]
                if feature_num is not None and vec_size is not None:
                    self.data_buffer.append(
                        np.reshape(datas, newshape=(channel, feature_num, vec_size)))
                else:
                    self.data_buffer.append(datas)
                self.label_buffer.append(int(data[label_key]))
                self.cnt += 1
        self.index = 0

    def next_batch(self, batch_size):
        data_batch = []
        label_batch = []
        for _ in range(batch_size):
            data_batch.append(self.data_buffer[self.index])
            label_batch.append(self.label_buffer[self.index])
            self.index = (self.index + 1) % self.cnt
        return np.array(data_batch), np.array(label_batch)

    def next(self, batch_size):
        return self.next_batch(batch_size)


class TextIterator(originIterator):
    def __init__(self, w2id, location, sentence_size, data_key='x', label_key='z'):
        self.cnt = 0
        self.data_buffer = []
        self.label_buffer = []
        self.sentence_size = sentence_size
        with open(location) as fin:
            for line in tqdm(fin):
                data = json.loads(line, encoding='utf8')
                self.data_buffer.append([w2id[w] for w in data[data_key].split() if w in w2id][:sentence_size])
                self.label_buffer.append(data[label_key])
                self.cnt += 1
        self.index = 0
        self.pad_id = w2id['</s>']

    def pad(self, sentences):
        for i in range(len(sentences)):
            len_ = len(sentences[i])
            sentences[i] += [self.pad_id] * (self.sentence_size - len_)

    def next_batch(self, batch_size):
        data_batch = []
        label_batch = []
        for _ in range(batch_size):
            data_batch.append(self.data_buffer[self.index])
            label_batch.append(self.label_buffer[self.index])
            self.index = (self.index + 1) % self.cnt
        self.pad(data_batch)
        return np.array(data_batch), np.array(label_batch)

    def next(self, batch_size):
        return self.next_batch(batch_size)


def extract_list(lists):
    return lists


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
                    if self.index % 100 == 0:
                        print(self.index)
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
