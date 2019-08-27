# coding: utf-8
# create by tongshiwei on 2019-8-27

__all__ = ["Indicesor"]

import random
from copy import deepcopy


class Indicesor(object):
    def __init__(self, key_iterable=None, shuffle=False, *args, **kwargs):
        if key_iterable is None:
            self.indices = None
        else:
            self.indices = self.initial_indices(key_iterable)
            if shuffle is True:
                random.shuffle(self.indices)

    def __call__(self, key_iterable=None, *args, **kwargs):
        if key_iterable is None:
            assert self.indices is not None
            return self.indices
        else:
            indices = self.initial_indices(key_iterable)
            random.shuffle(indices)
            return indices

    def initial_indices(self, key_iterable):
        if hasattr(key_iterable, "__len__"):
            indices = range(len(key_iterable))
        else:
            indices = None
            for idx, _ in enumerate(deepcopy(key_iterable)):
                indices = idx
            assert indices is not None
            indices = list(range(indices))
        self.indices = indices
        return self.indices

    @staticmethod
    def get_indices(key_iterable, shuffle=False):
        return Indicesor(key_iterable, shuffle=shuffle).indices
