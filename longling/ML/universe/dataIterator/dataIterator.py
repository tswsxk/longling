# coding: utf-8
# create by tongshiwei on 2017/12/6

from abc import ABCMeta, abstractmethod


class OriginIterator(object):
    __metaclass__ = ABCMeta

    def tolist(self):
        listbuff = []
        for elem in self:
            listbuff.append(elem)
        return listbuff

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    @abstractmethod
    def next(self):
        pass


class OriginBatchIterator(OriginIterator):
    def tolist(self):
        listbuff = []
        for elems in self:
            listbuff.extend(elems)
        return listbuff

    @abstractmethod
    def next_batch(self):
        pass

    def next(self):
        return self.next_batch()
