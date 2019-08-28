# coding:utf-8
# created by tongshiwei on 2018/10/9

import math
import warnings

import numpy
from longling.ML.toolkit.dataset.splitter.splitter import Splitter
from tqdm import tqdm

__all__ = [
    "RatioSplitter", "TTSplitter", "TVSplitter", "TVTSplitter",
    "KFoldSplitter"

]


class RatioSplitter(Splitter):
    """Splitter for multi ratios, using uniform sampling"""

    def __init__(self, key_iterable, *ratios, shuffle_indices=False, silent=False):
        super(RatioSplitter, self).__init__(key_iterable, shuffle_indices=shuffle_indices, silent=silent)

        ratios = list(ratios)

        ratios[-1] = 1.0 - sum(ratios[:-1]) if ratios[-1] is None else ratios[-1]

        assert isinstance(all(ratios), float), "all of ratios should be float, now are %s" % ratios
        assert 0.0 <= all(ratios) <= 1.0, "all of ratios should be in [0.0, 1.0], now are %s" % ratios

        ratios_sum = sum(ratios)
        if ratios_sum <= 1:
            warnings.warn("the sum of ratios %s is %s, little than 1.0" % (ratios, ratios_sum))
        elif ratios_sum < 0.0 or ratios_sum > 1.0:
            raise ValueError("the sum of ratios %s is %s, not in [0.0, 1.0]" % (ratios, ratios_sum))

        cum_ratios = numpy.cumsum([0.0] + ratios)
        separator_indices = [
            math.ceil(len(self.indices) * ratio) for ratio in cum_ratios
        ]
        self.separator_indices = [
            set(self.indices[separator_indices[i]: separator_indices[i + 1]])
            for i in range(len(separator_indices) - 1)
        ]

    def _split(self, source, target):
        assert len(target) == len(self.separator_indices)
        for idx, elem in tqdm(enumerate(source), "split %s" % source, disable=self.silent):
            for i, separated_indices in enumerate(self.separator_indices):
                if idx in separated_indices:
                    target[i].add(elem)


class TTSplitter(RatioSplitter):
    """Splitter for train and test, using uniform sampling"""

    def __init__(self, key_iterable, train_ratio=0.8, test_ratio=0.2, shuffle_indices=False, **kwargs):
        super(TTSplitter, self).__init__(
            key_iterable, train_ratio, test_ratio, shuffle_indices=shuffle_indices, **kwargs
        )


class TVSplitter(RatioSplitter):
    """Splitter for train and valid, using uniform sampling"""

    def __init__(self, key_iterable, train_ratio=0.8, valid_ratio=0.2, shuffle_indices=False, **kwargs):
        super(TVSplitter, self).__init__(
            key_iterable, train_ratio, valid_ratio, shuffle_indices=shuffle_indices, **kwargs
        )


class TVTSplitter(RatioSplitter):
    """Splitter for train valid and test, using uniform sampling"""

    def __init__(self, key_iterable, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, shuffle_indices=False, **kwargs):
        super(TVTSplitter, self).__init__(
            key_iterable, train_ratio, valid_ratio, test_ratio, shuffle_indices=shuffle_indices, **kwargs
        )


class KFoldSplitter(Splitter):
    def __init__(self, key_iterable, n_splits, silent=False):
        super(KFoldSplitter, self).__init__(key_iterable, shuffle_indices=False, silent=silent)
        sample_num = len(self.indices)
        proportion = sample_num / n_splits

        step = math.floor(proportion * sample_num)
        self.indices_buckets = [
            (i, i + step) for i in range(0, sample_num, step)
        ]

    def _split(self, source, target):
        assert len(target) == len(self.indices_buckets)
        for idx, elem in tqdm(enumerate(source), "split %s" % source, disable=self.silent):
            for i, (start, end) in enumerate(self.indices_buckets):
                if start <= idx < end:
                    target[i][0].add(elem)
                else:
                    target[i][1].add(elem)
