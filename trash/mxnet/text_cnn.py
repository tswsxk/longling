# coding:utf-8
# created by tongshiwei on 2018/7/6


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
from trash.io import VecDict
from longling.framework.ML.MXnet.monitor import TimeMonitor
from longling.framework.ML.MXnet.sym_lib import text_cnn_cell

from longling.lib.utilog import config_logging


def text_cnn():
    ############################################################################
    # parameters config
    logging.getLogger().setLevel(logging.INFO)
    propagate = False

    root = "../../../../../"

    dict_file = root + "data/word2vec/comment.vec.dat"
    train_file = root + "data/text/mini.instance.train"
    test_file = root + "data/text/mini.instance.test"

    model_dir = root + "data/mxnet/text_cnn/"
    model_name = "text_cnn"

    sentence_size = 30
    batch_size = 128
    last_batch_handle = 'pad'

    begin_epoch = 0
    num_epoch = 10
    ctx = mx.cpu()

    module_logger = {
        "filename": None,
        "logger": model_name,
        "console_log_level": logging.INFO,
        "propagate": False,
    }
    module_logger = config_logging(**module_logger)
    train_logger = config_logging(logger="train", console_log_level=logging.INFO, propagate=propagate)
    validation_logger = config_logging(
        filename=model_dir + "result.log",
        logger="validation",
        mode="w",
        log_format="%(message)s",
        propagate=propagate,
    )
    validation_result_file = model_dir + "result"

    ############################################################################

    ############################################################################
    # loading data
    w2v_dict = VecDict(dict_file)
    vocab_size, vec_size = w2v_dict.vocab_size, w2v_dict.vec_size
    embedding = w2v_dict.embedding

    def get_iter(filename):
        datas = []
        labels = []
        with open(filename) as f:
            for line in tqdm(f, desc="reading file[%s]" % filename):
                if not line.strip():
                    continue
                data = json.loads(line)
                datas.extend(w2v_dict.s2i([data['x']], sentence_size=sentence_size, sentence_split=True).tolist())
                labels.append(int(data['z']))

        return mx.io.NDArrayIter(
            data=np.asarray(datas), label=np.asarray(labels),
            batch_size=batch_size, last_batch_handle=last_batch_handle,
            data_name='data', label_name='label',
        )

    train_data = get_iter(train_file)
    test_data = get_iter(test_file)

    ############################################################################

    ############################################################################
    # network building
    def sym_gen(filter_list, num_filter, vocab_size, vec_size, sentence_size, num_label):
        data = mx.symbol.Variable('data')
        embed_weight = mx.symbol.Variable('word_embedding')
        data = mx.symbol.expand_dims(data, axis=1)
        data = mx.symbol.Embedding(data, weight=embed_weight, input_dim=vocab_size,
                                   output_dim=vec_size, name="word_embedding")
        data = mx.sym.BlockGrad(data)
        label = mx.symbol.Variable('label')
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
        )
        sym = mx.symbol.SoftmaxOutput(
            data=data,
            label=label,
        )
        return sym

    sym = sym_gen(
        filter_list=[1, 2, 3, 4],
        num_filter=60,
        vocab_size=vocab_size,
        vec_size=vec_size,
        sentence_size=sentence_size,
        num_label=2,
    )
    mod = mx.mod.Module(
        symbol=sym,
        data_names=['data', ],
        label_names=['label', ],
        logger=module_logger,
        context=ctx,
    )

    plot_network(
        nn_symbol=sym,
        save_path=model_dir + "plot/network",
        shape=form_shape(train_data),
        node_attrs={"fixedsize": "false"},
        view=True
    )
    ############################################################################

    ############################################################################
    # fitting
    time_monitor = TimeMonitor()
    tins = tqdm_speedometer(begin_epoch)
    cross_entropy = CrossEntropy()
    speedometer = Speedometer(
        batch_size=batch_size,
        frequent=100,
        metrics=cross_entropy,
        logger=train_logger,
    )

    mod.fit(
        train_data=train_data,
        eval_data=test_data,
        eval_metric=[PRF(), Accuracy(), cross_entropy],
        eval_end_callback=ClassificationLogValidationMetricsCallback(
            time_monitor,
            logger=validation_logger,
            logfile=validation_result_file,
        ),
        # eval_batch_end_callback=[vd_logger1],
        begin_epoch=begin_epoch,
        num_epoch=num_epoch,
        optimizer='rmsprop',
        arg_params={'word_embedding': mx.nd.array(embedding)},
        allow_missing=True,
        optimizer_params={'learning_rate': 0.0005},
        batch_end_callback=[speedometer, tins],
        epoch_end_callback=[
            mx.callback.do_checkpoint(model_dir + model_name),
            TqdmEpochReset(tins, "training"),
        ],
        monitor=time_monitor,
    )
    ############################################################################

    ############################################################################
    # clean
    tins.close()
    ############################################################################
