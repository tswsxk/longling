# coding: utf-8
# create by tongshiwei on 2019/4/13

from mxnet.gluon.nn import Embedding
from mxnet.ndarray import NDArray

__all__ = ["set_embedding_weight"]


def set_embedding_weight(embedding_layer, weight):
    """
    Parameters
    ----------
    embedding_layer: Embedding
    weight: NDArray
    """
    embedding_layer.weight.set_data(weight)
