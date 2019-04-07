# coding:utf-8
# created by tongshiwei on 2018/7/6

from __future__ import absolute_import

import logging

import mxnet as mx


from longling.framework.ML.MXnet.callback import tqdm_speedometer, \
    ClassificationLogValidationMetricsCallback, Speedometer, \
    TqdmEpochReset
from longling.framework.ML.MXnet.metric import PRF, Accuracy, CrossEntropy
from longling.framework.ML.MXnet.viz import plot_network, form_shape
from trash.io import DictJsonIter
from longling.framework.ML.MXnet.monitor import TimeMonitor
from longling.framework.ML.MXnet.sym_lib import dnn_cell

from longling.lib.utilog import config_logging


def dnn():
    ############################################################################
    # parameters config

    # file path
    root = "../../../../../"
    train_file = root + "data/image/one_dim/mnist_train"
    test_file = root + "data/image/one_dim/mnist_test"

    model_dir = root + "data/mxnet/dnn/"
    model_name = "dnn"

    # model args
    batch_size = 128
    last_batch_handle = 'pad'

    num_hiddens = [128, 64, 32]
    num_label = 10

    begin_epoch = 0
    num_epoch = 10
    ctx = mx.cpu()

    # infoer
    propagate = False
    module_logger = {
        "filename": None,
        "logger": model_name,
        "console_log_level": logging.INFO,
        "propagate": False,
    }
    module_logger = config_logging(**module_logger)
    train_logger = config_logging(logger="train", console_log_level=logging.INFO, propagate=propagate)
    validation_logger = config_logging(
        logger="validation",
        filename=model_dir + "result.log",
        mode="w",
        console_log_level=logging.INFO,
        log_format="%(message)s",
        propagate=propagate,
    )
    validation_result_file = model_dir + "result"
    ############################################################################

    ############################################################################
    # loading data
    data_key_dict = {'data': 'x'}
    label_key_dict = {'label': 'z'}

    def get_iter(filename):
        return DictJsonIter(
            filename=filename,
            batch_size=batch_size,
            data_key_dict=data_key_dict,
            label_key_dict=label_key_dict,
            last_batch_handle=last_batch_handle,
        )

    train_data = get_iter(train_file)
    test_data = get_iter(test_file)

    ############################################################################

    ############################################################################
    # network building
    def sym_gen(num_hiddens, num_label):
        data = mx.symbol.Variable('data')
        label = mx.symbol.Variable('label')
        data = dnn_cell(
            in_sym=data,
            num_hiddens=num_hiddens,
            num_output=num_label,
            dropout=0.5,
            inner_dropouts=0.5,
            batch_norms=1,
        )
        sym = mx.symbol.SoftmaxOutput(
            data=data,
            label=label,
        )
        return sym

    sym = sym_gen(num_hiddens, num_label)
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
    tins = tqdm_speedometer()
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
        begin_epoch=begin_epoch,
        num_epoch=num_epoch,
        optimizer='rmsprop',
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
