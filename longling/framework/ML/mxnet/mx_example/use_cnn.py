# coding:utf-8
# created by tongshiwei on 2018/7/6

from __future__ import absolute_import


import json

import mxnet as mx
import numpy as np


def cnn_use():
    root = "../../../../../"
    model_dir = root + "data/mxnet/cnn/"
    model_name = "cnn"

    epoch = 20

    prefix = model_dir + model_name
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    internals = sym.get_internals()

    fc = internals['fullyconnected1_output']
    loss = mx.symbol.MakeLoss(mx.symbol.max(fc, axis=1))
    conv0 = internals['convolution0_output']
    conv1 = internals['convolution1_output']

    # group_output = mx.symbol.Group([loss, conv0, conv1])
    model = mx.mod.Module(
        symbol=loss,
        context=mx.cpu(),
        data_names=['data', ],
        label_names=[],
    )

    # model._arg_params = arg_params
    # model._aux_params = aux_params
    # model.params_initialized = True

    batch_size = 1

    from collections import namedtuple

    Batch = namedtuple('Batch', ['data'])

    skip = 50
    with open(root + "data/image/mnist_test") as f:
        for _ in range(skip):
            f.readline()
        data = f.readline()
        data = json.loads(data)['x']
    datas = [data]
    datas = Batch([mx.nd.array(datas)])

    model.bind(
        data_shapes=[('data', (batch_size, 1, 28, 28))],
        force_rebind=True,
        for_training=True,
        inputs_need_grad=True,
    )
    model.set_params(arg_params, aux_params, allow_missing=True)

    model.forward(datas, is_train=True)
    model.backward()

    # value = model.get_outputs()
    #
    # labels, conv1, conv2 = value[0].asnumpy(), value[1].asnumpy(), value[2].asnumpy()
    # labels = labels.argmax(axis=1)
    # conv1 = np.average(conv1, axis=1)
    # conv1 = (conv1 - conv1.min()) / (conv1.max() - conv1.min()) * 256
    # conv2 = np.average(conv2, axis=1)
    # conv2 = (conv2 - conv2.min()) / (conv2.max() - conv2.min()) * 256
    ig = model.get_input_grads()[0].asnumpy()[0][0]
    ig = (ig - ig.min()) / (ig.max() - ig.min()) * 256

    from PIL import Image
    for idx, data in enumerate(datas.data):
        im0 = Image.fromarray(np.uint8(data.asnumpy()[0][0] * 256), mode='L')
        im0 = im0.resize((256, 256), Image.ANTIALIAS)
        im0.show()
        # im1 = Image.fromarray(conv1[idx], mode='L')
        # im1 = im1.resize((512, 512))
        # im1.show()
        # im2 = Image.fromarray(conv2[idx], mode='L')
        # im2 = im2.resize((256, 256))
        # im2.show()
        im3 = Image.fromarray(np.uint8(ig), mode='L')
        im3 = im3.resize((256, 256), Image.ANTIALIAS)
        im3.show()
