# coding: utf-8
# create by tongshiwei on 2019-8-27

__all__ = ["Indicesor"]

import io
import random
from copy import deepcopy
from tqdm import tqdm
from longling import config_logging

logger = config_logging(logger="longling")


class Indicesor(object):
    def __init__(self, key_iterable=None, shuffle=False, *args, silent=False, **kwargs):
        self.silent = silent
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
            def get_indices(_key_iterable):
                _indices = None
                for idx, _ in tqdm(enumerate(_key_iterable), "initial indices", disable=self.silent):
                    _indices = idx
                assert _indices is not None
                return list(range(_indices))

            if isinstance(key_iterable, io.TextIOWrapper):
                offset = key_iterable.tell()
                indices = get_indices(key_iterable)
                key_iterable.seek(offset)
            else:
                try:
                    key_iterable = deepcopy(key_iterable)
                except TypeError as e:
                    logger.warning("Skip deepcopy procedure")
                    logger.warning(e)
                indices = get_indices(key_iterable)
        self.indices = indices
        return self.indices

    @staticmethod
    def get_indices(key_iterable, shuffle=False, silent=False):
        return Indicesor(key_iterable, shuffle=shuffle, silent=silent).indices
