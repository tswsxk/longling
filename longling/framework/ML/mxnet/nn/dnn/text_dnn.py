# coding: utf-8
import logging

import mxnet as mx

from longling.framework.ML.mxnet.nn.shared.nn import get_model, get_embeding_model


def get_text_dnn_symbol_without_loss(vocab_size, vec_size, num_hiddens=[100], num_label=2, dropout=0.0):
    data = mx.sym.Variable('data')

    net = mx.sym.Embedding(data=data, input_dim=vocab_size, output_dim=vec_size, name='embedding')
    net = mx.sym.sum(net, axis=1, name="sum")

    for i in xrange(len(num_hiddens)):
        net = mx.sym.FullyConnected(data=net, name='fc%s' % i, num_hidden=num_hiddens[i])
        net = mx.sym.Activation(data=net, name='relu%s' % i, act_type="relu")
    if dropout > 0.0:
        net = mx.sym.Dropout(data=net, p=dropout)
    fc = mx.sym.FullyConnected(data=net, name='fc', num_hidden=num_label)
    return fc


def get_text_dnn_symbol(vocab_size, vec_size, num_hiddens=[100], num_label=2, dropout=0.0):
    label = mx.sym.Variable('label')
    fc = get_text_dnn_symbol_without_loss(
        vocab_size=vocab_size,
        vec_size=vec_size,
        num_hiddens=num_hiddens,
        num_label=num_label,
        dropout=dropout,
    )
    sm = mx.sym.SoftmaxOutput(data=fc, label=label, name='softmax')
    return sm


def get_text_dnn_model(ctx, dnn_symbol, embedding, sentence_size, batch_size, return_grad=False,
                       checkpoint=None, logger=logging):
    return get_embeding_model(
        ctx=ctx,
        nn_symbol=dnn_symbol,
        embedding=embedding,
        feature_num=sentence_size,
        batch_size=batch_size,
        return_grad=return_grad,
        checkpoint=checkpoint,
        logger=logger,
    )


def get_text_dnn_model_without_embeding(ctx, dnn_symbol, vec_size, sentence_size, batch_size, return_grad=False,
                                        checkpoint=None, logger=logging):
    return get_model(
        ctx=ctx,
        nn_symbol=dnn_symbol,
        vec_size=vec_size,
        feature_num=sentence_size,
        batch_size=batch_size,
        return_grad=return_grad,
        checkpoint=checkpoint,
        logger=logger,
    )
