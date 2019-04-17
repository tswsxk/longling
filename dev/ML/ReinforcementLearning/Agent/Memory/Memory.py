# coding:utf-8
# created by tongshiwei on 2018/11/8
import random
import logging


class Memory(object):
    def __init__(self, max_size=1000000, logger=logging, *args, **kwargs):
        self._memory = []
        self.max_size = max_size
        self.logger = logger
        self.top = 0

    def add(self, *args, **kwargs):
        raise NotImplementedError

    def get_memory(self, size):
        if size > len(self._memory):
            self.logger.warning("requiring size bigger than the memory, only return %s" % len(self._memory))
            return self._memory
        else:
            return random.sample(self._memory, size)

    def get(self, size):
        return self.get_memory(size)

    def reset(self):
        self._memory = []

    @property
    def size(self):
        return len(self._memory)

