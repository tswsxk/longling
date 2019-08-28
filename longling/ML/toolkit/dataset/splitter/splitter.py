# coding: utf-8
# create by tongshiwei on 2019-8-27

__all__ = ["Splitter", "split", "divide"]

from ..indicesor import Indicesor


class Splitter(object):
    def __init__(self, key_iterable, shuffle_indices=False, *args, silent=False, **kwargs):
        self.indices = Indicesor.get_indices(key_iterable, shuffle=shuffle_indices, silent=silent)
        self.silent = silent

    def _split(self, source, target):
        raise NotImplementedError

    def split(self, *source_target_iterable):
        for source, target in source_target_iterable:
            self._split(source, target)

    def __call__(self, *source_target_iterable):
        return self.split(*source_target_iterable)


def split(splitter, *source_target_iterable):
    splitter(*source_target_iterable)


def divide(splitter_class, *source_target_iterable, **splitter_params):
    splitter = splitter_class(**splitter_params)
    split(splitter, *source_target_iterable)
