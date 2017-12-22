# coding: utf-8
from __future__ import unicode_literals

import logging

import mxnet as mx

from longling.framework.ML.mxnet.nn.shared.nn import get_model


def get_numerical_dnn_symbol_without_loss(num_hiddens=[100], num_label=2, dropout=0.0):
    data = mx.sym.Variable('data')

    net = data

    for i in xrange(len(num_hiddens)):
        net = mx.sym.FullyConnected(data=net, name='fc%s' % i, num_hidden=num_hiddens[i])
        # net = mx.sym.Activation(data=net, name='relu%s' % i, act_type="relu")
    if dropout > 0.0:
        net = mx.sym.Dropout(data=net, p=dropout)
    fc = mx.sym.FullyConnected(data=net, name='fc', num_hidden=num_label)
    return fc


def get_numerical_dnn_symbol(num_hiddens=[100], num_label=2, dropout=0.0):
    label = mx.sym.Variable('label')
    fc = get_numerical_dnn_symbol_without_loss(
        num_hiddens=num_hiddens,
        num_label=num_label,
        dropout=dropout,
    )
    sm = mx.sym.SoftmaxOutput(data=fc, label=label, name='softmax')
    return sm


def get_text_dnn_symbol(num_hiddens=[100], num_label=2, dropout=0.0):
    label = mx.sym.Variable('label')
    fc = get_numerical_dnn_symbol_without_loss(
        num_hiddens=num_hiddens,
        num_label=num_label,
        dropout=dropout,
    )
    sm = mx.sym.SoftmaxOutput(data=fc, label=label, name='softmax')
    return sm


def get_numerical_dnn_model(ctx, dnn_symbol, feature_num, batch_size, vec_size=-1, channel=-1, return_grad=False,
                            checkpoint=None, logger=logging):
    return get_model(
        ctx=ctx,
        nn_symbol=dnn_symbol,
        channel=channel,
        feature_num=feature_num,
        vec_size=vec_size,
        batch_size=batch_size,
        return_grad=return_grad,
        checkpoint=checkpoint,
        logger=logger,
    )

def grad_verify():
    from longling.framework.ML.mxnet.nn.shared.nn import NN
    symbol = get_numerical_dnn_symbol_without_loss(
        [],
    )
    symbol = mx.symbol.MakeLoss(mx.symbol.max(symbol, axis=1))
    NN.plot_network(symbol, shape={'data': (128, 3)}, show_tag=False)
    model = get_numerical_dnn_model(
        ctx=mx.cpu(0),
        dnn_symbol=symbol,
        feature_num=1,
        batch_size=2,
        return_grad=True,
    )
    import numpy as np
    x = 10
    dx = 0.0000001 * x
    model.data[:] = np.array([[x], [x + dx]], dtype=np.float64)
    y = model.model_exec.forward(is_train=False)
    print (y[0][1] - y[0][0]) / dx
    model.model_exec.backward()
    if 1 == 1:
        pass

if __name__ == '__main__':
    pass
