# coding: utf-8
# create by tongshiwei on 2017/10/23
from abc import ABCMeta, abstractmethod
import json
import logging
import os

import numpy as np

from tqdm import tqdm


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
