# coding:utf-8
# created by tongshiwei on 2018/7/11
from __future__ import absolute_import
from __future__ import print_function


import mxnet as mx
import numpy as np
from mxnet import nd, gluon


def cnn_use():
    root = "../../../../"

    model_dir = root + "data/gluon/cnn/"
    model_name = "cnn"

    epoch = 10

    def transform(data, label):
        return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)

    batch_size = 128

    filename = model_dir + model_name + "-%04d.parmas" % epoch

    model = nd.load(filename)
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(model)
    test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                         batch_size, shuffle=False)
    for data, label in test_data:
        output = net(data)
        break

    mx.gluon.model_zoo.vision.alexnet()