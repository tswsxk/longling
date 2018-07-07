# coding: utf-8
# created by tongshiwei on 18-1-27


from __future__ import absolute_import

import json
import logging

import mxnet as mx
import numpy as np

from tqdm import tqdm

from longling.framework.ML.MXnet.callback import tqdm_speedometer, \
    ClassificationLogValidationMetricsCallback, Speedometer, \
    TqdmEpochReset
from longling.framework.ML.MXnet.metric import PRF, Accuracy, CrossEntropy
from longling.framework.ML.MXnet.viz import plot_network, form_shape
from longling.framework.ML.MXnet.io_lib import DictJsonIter, VecDict, SimpleBucketIter
from longling.framework.ML.MXnet.monitor import TimeMonitor
from longling.framework.ML.MXnet.sym_lib import lenet_cell, text_cnn_cell
from longling.framework.ML.MXnet.util import get_fine_tune_model

from longling.lib.utilog import config_logging

def text_cnn():
    vocab_size = 365000
    vec_size = 80
    sentence_size = 25
    root = "../../../../../"
    model_dir = root + "data/mxnet/cnn/"
    model_name = "cnn"


    ############################################################################
    # network building
    def sym_gen(filter_list, num_filter, vocab_size, vec_size, sentence_size, num_label, margin):
        data1 = mx.symbol.Variable('data1')
        data2 = mx.symbol.Variable('data2')
        embed_weight = mx.symbol.Variable('word_embedding')
        # margin = mx.sym.constant(margin)

        def sub_sym_gen(data):
            data = mx.symbol.expand_dims(data, axis=1)
            data = mx.symbol.Embedding(data, weight=embed_weight, input_dim=vocab_size,
                                       output_dim=vec_size, name="word_embedding")
            data = mx.sym.BlockGrad(data)
            data = text_cnn_cell(
                in_sym=data,
                sentence_size=sentence_size,
                vec_size=80,
                num_output=num_label,
                filter_list=filter_list,
                num_filter=num_filter,
                dropout=0.5,
                batch_norms=1,
                highway=True,
                name_prefix="text_cnn",
            )
            data = mx.sym.softmax(data)
            return data

        data1 = sub_sym_gen(data1)
        data2 = sub_sym_gen(data2)
        # pairloss = mx.sym.min(mx.ndarray.array([0]), (margin + data1 - data2))
        # sym = mx.symbol.make_loss(pairloss)
        sym = mx.sym.concat(data1, data2)
        print(data2.list_arguments())
        print(data1.list_arguments())
        # print(sym.list_arguments())
        return sym

    sym = sym_gen(
        filter_list=[1, 2, 3, 4],
        num_filter=60,
        vocab_size=vocab_size,
        vec_size=vec_size,
        sentence_size=sentence_size,
        num_label=2,
        margin=0.2,
    )

    plot_network(
        nn_symbol=sym,
        save_path=model_dir + "plot/network",
        node_attrs={"fixedsize": "false"},
        view=True
    )
    ############################################################################

# todo
# plot data on website
# implement Jin experiment(r-cnn)
# GAN
# poem rnn
# picture to poem
# rcnn review
# rl network
# mxnet imgae iter

if __name__ == '__main__':
    text_cnn()