# coding: utf-8
# created by tongshiwei on 18-1-27

import mxnet as mx


def get_name(name_prefix=None, name=""):
    if name_prefix is None:
        return None
    return name_prefix + "_" + name


def dnn_cell(in_sym, num_hiddens=[100], num_output=2, dropout=0.0, inner_dropouts=None, batch_norms=0):
    net = in_sym

    if inner_dropouts is None:
        dropouts = [0.0] * (len(num_hiddens) - 1) + [dropout]

    else:
        if isinstance(inner_dropouts, list):
            if len(inner_dropouts) == len(num_hiddens):
                dropouts = inner_dropouts
            elif len(inner_dropouts) == len(num_hiddens) - 1:
                dropouts = inner_dropouts + [dropout]
            else:
                raise Exception()
        else:
            dropouts = [inner_dropouts] * (len(num_hiddens) - 1) + [dropout]

    assert len(dropouts) == len(num_hiddens)

    if batch_norms == 0:
        batch_norms = [0] * len(num_hiddens)

    elif batch_norms == 1:
        batch_norms = [1] * len(num_hiddens)

    else:
        assert isinstance(batch_norms, list)

        if len(batch_norms) == len(num_hiddens):
            batch_norms += [0]

        assert len(batch_norms) == len(num_hiddens)

    for i in range(len(num_hiddens)):
        net = mx.sym.FullyConnected(data=net, num_hidden=num_hiddens[i])
        net = mx.sym.Activation(data=net, act_type="relu")
        if batch_norms[i] == 1:
            net = mx.symbol.BatchNorm(net)
        if dropouts[i] > 0.0:
            net = mx.sym.Dropout(data=net, p=dropouts[i])

    fc = mx.sym.FullyConnected(data=net, num_hidden=num_output)
    return fc


def lenet_cell(in_sym, num_output=2):
    net = in_sym

    # first conv layer
    conv1 = mx.sym.Convolution(data=net, kernel=(5, 5), num_filter=20)
    tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
    pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2, 2), stride=(2, 2))
    # second conv layer
    conv2 = mx.sym.Convolution(data=pool1, kernel=(5, 5), num_filter=50)
    tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
    pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2, 2), stride=(2, 2))
    # first fullc layer
    flatten = mx.sym.flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")

    net = tanh3

    fc = mx.sym.FullyConnected(data=net, num_hidden=num_output)
    return fc


def text_cnn_cell(in_sym, sentence_size, vec_size, num_output=2, filter_list=[1, 2, 3, 4], num_filter=60, dropout=0.0,
                  batch_norms=0, highway=False, name_prefix=None):
    pooled_outputs = []

    for i, filter_size in enumerate(filter_list):
        convi = mx.sym.Convolution(
            data=in_sym,
            kernel=(filter_size, vec_size),
            num_filter=num_filter,
            name=get_name(name_prefix, "convolution%s" % i),
        )
        relui = mx.sym.Activation(data=convi, act_type='relu')
        pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(sentence_size - filter_size + 1, 1), stride=(1, 1))
        if batch_norms > 0:
            pooli = mx.sym.BatchNorm(pooli, name=get_name(name_prefix, "batchnorm%s" % i),)
        pooled_outputs.append(pooli)

    total_filters = num_filter * len(filter_list)
    concat = mx.sym.Concat(dim=1, *pooled_outputs)
    h_pool = mx.sym.Reshape(data=concat, shape=(-1, total_filters))
    if highway:
        h_pool = highway_cell(h_pool, len(filter_list) * num_filter, name_prefix=get_name(name_prefix, "highway"))
    if dropout > 0.0:
        h_drop = mx.sym.Dropout(data=h_pool, p=dropout)
    else:
        h_drop = h_pool
    fc = mx.sym.FullyConnected(data=h_drop, num_hidden=num_output, name=get_name(name_prefix, "fc"))

    return fc


def highway_cell(data, num_hidden, name_prefix=None):
    _data = data
    high_fc = mx.sym.FullyConnected(
        data=data,
        num_hidden=num_hidden,
        name=get_name(name_prefix, "fullyconnected0")
    )
    high_relu = mx.sym.Activation(high_fc, act_type='relu')

    high_trans_fc = mx.sym.FullyConnected(
        data=_data,
        num_hidden=num_hidden,
        name=get_name(name_prefix, "fullyconnected1")
    )
    high_trans_sigmoid = mx.sym.Activation(high_trans_fc, act_type='sigmoid')

    return high_relu * high_trans_sigmoid + _data * (1 - high_trans_sigmoid)

