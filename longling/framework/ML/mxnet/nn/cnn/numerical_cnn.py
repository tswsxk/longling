# coding:utf-8
import logging

import mxnet as mx

from longling.framework.ML.mxnet.nn.shared.nn import get_model


def get_numerical_cnn_symbol_without_loss(batch_size, feature_num, vec_size, channel=1, num_label=2,
                                          kernel_list=[[(5, 5), (5, 5)]],
                                          pool_list=[[(2, 2), (2, 2)]], fc_list=[[84]], num_filter_list=[[1, 1]],
                                          dropout=0.0):
    input_x = mx.sym.Variable('data')

    conv_input = mx.sym.Reshape(data=input_x, shape=(batch_size, channel, feature_num, vec_size))

    outputs = []
    for i, kernel_sizes in enumerate(kernel_list):
        pool_sizes = pool_list[i]
        fc_sizes = fc_list[i]
        num_filters = num_filter_list[i]
        layer = conv_input
        for j, kernel_size in enumerate(kernel_sizes):
            convi = mx.sym.Convolution(data=layer, kernel=kernel_size, num_filter=num_filters[j],
                                       name=('convolution_%s_%s' % (i, j)))
            relui = mx.sym.Activation(data=convi, act_type='relu', name=('activation%s_%s' % (i, j)))
            if j < len(pool_sizes):
                layer = mx.sym.Pooling(data=relui, pool_type='max', kernel=pool_sizes[j],
                                       stride=pool_sizes[j], name=('pooling_%s_%s' % (i, j)))
            else:
                layer = relui

        concat = mx.sym.Concat(dim=1, *layer)
        if dropout > 0.0:
            h_drop = mx.sym.Dropout(data=concat, p=dropout)
        else:
            h_drop = concat

        layer = h_drop
        for j, fc_num in enumerate(fc_sizes):
            fc_weight = mx.sym.Variable('fc_%s_%s_weight' % (i, j))
            fc_bias = mx.sym.Variable('fc_%s_%s_bias' % (i, j))
            layer = mx.sym.FullyConnected(data=layer, weight=fc_weight, bias=fc_bias, num_hidden=fc_num)

        outputs.append(layer)

    concat = mx.sym.Concat(dim=1, *outputs)
    if dropout > 0.0:
        h_drop = mx.sym.Dropout(data=concat, p=dropout)
    else:
        h_drop = concat

    cls_weight = mx.sym.Variable('cls_weight')
    cls_bias = mx.sym.Variable('cls_bias')
    fc = mx.sym.FullyConnected(data=h_drop, weight=cls_weight, bias=cls_bias, num_hidden=num_label)

    return fc


def get_numerical_cnn_symbol(batch_size, feature_num, vec_size, channel=1, num_label=2, kernel_list=[[(5, 5), (5, 5)]],
                             pool_list=[[(2, 2), (2, 2)]], fc_list=[[84]], num_filter_list=[[1, 1]], dropout=0.0):
    input_y = mx.sym.Variable('label')
    fc = get_numerical_cnn_symbol_without_loss(
        batch_size=batch_size,
        channel=channel,
        feature_num=feature_num,
        vec_size=vec_size,
        num_label=num_label,
        kernel_list=kernel_list,
        pool_list=pool_list,
        fc_list=fc_list,
        num_filter_list=num_filter_list,
        dropout=dropout,
    )
    sm = mx.sym.SoftmaxOutput(data=fc, label=input_y, name='softmax')

    return sm


def get_numerical_cnn_model(ctx, cnn_symbol, feature_num, vec_size, batch_size, channel=1, return_grad=False,
                            checkpoint=None, logger=logging):
    return get_model(
        ctx=ctx,
        nn_symbol=cnn_symbol,
        vec_size=vec_size,
        feature_num=feature_num,
        channel=channel,
        batch_size=batch_size,
        return_grad=return_grad,
        checkpoint=checkpoint,
        logger=logger,
    )


# some typical cnn network
def lenet5like_symbol(batch_size, feature_num, vec_size, num_label=2, dropout=0.0,
                      kernel_list=[[(5, 5), (5, 5), (5, 5)]],
                      pool_list=[[(2, 2), (2, 2)]], fc_list=[[84]], num_filter_list=[[6, 16, 120]]):
    return get_numerical_cnn_symbol_without_loss(
        batch_size=batch_size,
        feature_num=feature_num,
        vec_size=vec_size,
        num_label=num_label,
        dropout=dropout,
        kernel_list=kernel_list,
        pool_list=pool_list,
        fc_list=fc_list,
        num_filter_list=num_filter_list,
    )
