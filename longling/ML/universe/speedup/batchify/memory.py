# coding:utf-8
# created by tongshiwei on 2018/7/26

from mxnet.gluon.data import DataLoader, ArrayDataset
from mxnet.gluon.data import SequentialSampler, RandomSampler, BatchSampler
from gluonnlp.data import FixedBucketSampler, SortedBucketSampler


def test_example():
    """
    >>> test_sentences = [
    ...     [1, 3, 5],
    ...     [2, 4],
    ...     [6],
    ...     [7, 8, 9],
    ...     [10, 1, 1, 1, 1],
    ...     [7, 8],
    ... ]
    >>> batch_size = 2
    >>> lengths = [len(sentence) for sentence in test_sentences]
    >>> batches = SortedBucketSampler(lengths, 2, reverse=False)
    >>> test_sentences[list(batches)[0][0]]
    [6]
    >>> batches = FixedBucketSampler(lengths, 2)
    >>> list(batches)
    [6]
    """
    pass

if __name__ == '__main__':
    import doctest
    doctest.testmod()

    test_example()