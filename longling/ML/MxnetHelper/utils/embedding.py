# coding: utf-8
# 2021/8/5 @ tongshiwei

from mxnet.gluon.nn import Embedding
from mxnet.ndarray import NDArray

__all__ = ["set_embedding_weight"]


def set_embedding_weight(embedding_layer, weight):
    """
    Parameters
    ----------
    embedding_layer: Embedding
    weight: NDArray

    Examples
    --------
    >>> import mxnet as mx
    >>> from mxnet import gluon
    >>> emb1 = gluon.nn.Embedding(2, 3)
    >>> emb1.initialize()
    >>> emb1.weight.set_data(mx.nd.ones((2, 3)))
    >>> emb1.weight.data()
    <BLANKLINE>
    [[1. 1. 1.]
     [1. 1. 1.]]
    <NDArray 2x3 @cpu(0)>
    >>> set_embedding_weight(emb1, mx.nd.zeros((2, 3)))
    >>> emb1.weight.data()
    <BLANKLINE>
    [[0. 0. 0.]
     [0. 0. 0.]]
    <NDArray 2x3 @cpu(0)>
    """
    embedding_layer.weight.set_data(weight)
