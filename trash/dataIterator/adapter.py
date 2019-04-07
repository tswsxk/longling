# coding: utf-8
# create by tongshiwei on 2017/12/6

from __future__ import absolute_import

from collections import OrderedDict

from longling.framework.ML.universe.dataIterator import OriginIterator, OriginBatchIterator


class Single2BatchAdapter(OriginBatchIterator):
    def __init__(self, single_iter, batch_size=1000):
        self.batch_size = batch_size
        self.input_iter = single_iter
        self.buff = []

    def next_batch(self):
        while True:
            try:
                elem = self.input_iter.next()
                self.buff.append(elem)
                if len(self.buff) == self.batch_size:
                    buff = self.buff
                    self.buff = []
                    assert buff
                    return buff
                elif len(self.buff) > self.batch_size:
                    raise Exception("self.buff is too big")
            except StopIteration:
                if self.buff:
                    buff = self.buff
                    self.buff = []
                    return buff
                raise StopIteration


class Batch2SingleAdapter(OriginIterator):
    def __init__(self, batch_iter):
        self.input_iter = batch_iter
        self.buff = []

    def next(self):
        if not self.buff:
            try:
                self.buff = self.input_iter.next()
            except StopIteration:
                raise StopIteration
        return self.buff.pop(0)


def iter_expand(iterator):
    expand_buff = None

    def _get_type(obj):
        if isinstance(obj, tuple):
            class ListBuff(object):
                def __init__(self, num):
                    self.buff = [[] for _ in range(num)]

                def add(self, data_obj):
                    for i, d in enumerate(data_obj):
                        self.buff[i].append(d)

            buff = ListBuff(len(obj))

        elif isinstance(obj, dict):
            class DictBuff(object):
                def __init__(self, keys):
                    self.buff = OrderedDict(zip(keys, [[] for _ in range(len(keys))]))

                def add(self, data_obj):
                    for k, v in data_obj.items():
                        self.buff[k].append(v)

            buff = DictBuff(obj.keys())

        else:
            buff = []

        return buff

    for data in iterator:
        if expand_buff is None:
            expand_buff = _get_type(data)
        expand_buff.add(data)

    return expand_buff
