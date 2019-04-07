# coding: utf-8
# created by tongshiwei on 18-2-6
import math

import mxnet as mx
from mxnet import gluon








if __name__ == '__main__':
    import random

    random.seed(10)

    batch_size = 3
    max_len = 10
    vec_dim = 5
    channel_size = 4
    length = [random.randint(1, max_len) for _ in range(128)]
    data = [[[[random.random() for _ in range(channel_size)] for _ in range(vec_dim)] for _ in range(max_len)] for l in
            length]

    data = mx.nd.array(data)
    print(data.shape)

    text_cnn = TextCNN(max_len, vec_dim, channel_size, 2)
    text_cnn.initialize()

    a = text_cnn(data)
    print(a.shape)
