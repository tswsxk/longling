# coding: utf-8
# create by tongshiwei on 2017/12/6

from __future__ import absolute_import

from longling.lib.dataIterator import originIterator, originBatchIterator


class single2batchAdapter(originBatchIterator):
    def __init__(self, singleIter, batchSize=1000):
        self.batchSize = batchSize
        self.inputIter = singleIter
        self.buff = []

    def next_batch(self):
        while True:
            try:
                elem = self.inputIter.next()
                self.buff.append(elem)
                if len(self.buff) == self.batchSize:
                    buff = self.buff
                    self.buff = []
                    assert buff
                    return buff
                elif len(self.buff) > self.batchSize:
                    raise Exception("self.buff is too big")
            except:
                if self.buff:
                    buff = self.buff
                    self.buff = []
                    return buff
                raise StopIteration


class batch2singleAdapter(originIterator):
    def __init__(self, batchIter):
        self.inputIter = batchIter
        self.buff = []

    def next(self):
        if not self.buff:
            try:
                self.buff = self.inputIter.next()
            except StopIteration:
                raise StopIteration
        return self.buff.pop(0)
