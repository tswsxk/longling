# coding:utf-8
# created by tongshiwei on 2018/7/6

from __future__ import absolute_import

import json
import logging

import mxnet as mx


from longling.framework.ML.mxnet.callback import tqdm_speedometer, \
    ClassificationLogValidationMetricsCallback, Speedometer, \
    TqdmEpochReset
from longling.framework.ML.mxnet.metric import PRF, Accuracy, CrossEntropy
from longling.framework.ML.mxnet.viz import plot_network, form_shape
from longling.framework.ML.mxnet.io_lib import DictJsonIter
from longling.framework.ML.mxnet.monitor import TimeMonitor
from longling.framework.ML.mxnet.util import get_fine_tune_model

from longling.lib.utilog import config_logging
from longling.framework.spider import url_download


def vgg16():
    ############################################################################
    # parameters config
    # logging.getLogger().setLevel(logging.INFO)
    propagate = False

    root = "../../../../../"
    train_file = root + "data/image/mnist_train"
    test_file = root + "data/image/mnist_test"

    model_dir = root + "data/mxnet/vgg16_mnist/"
    model_name = "vgg16_mnist"

    batch_size = 1
    last_batch_handle = 'pad'

    begin_epoch = 1
    num_epoch = 10
    ctx = mx.cpu()

    module_logger = {
        "filename": None,
        "logger": model_name,
        "console_log_level": logging.INFO,
        "propagate": True,
    }
    module_logger = config_logging(**module_logger)
    train_logger = config_logging(logger="train", console_log_level=logging.INFO, propagate=propagate)
    validation_logger = config_logging(
        logger="validation",
        filename=model_dir + "result.log",
        mode="w",
        console_log_level=logging.INFO,
        propagate=propagate,
    )
    validation_result_file = model_dir + "result"

    # pre-trained model parameters
    pre_trained_path = root + "data/mxnet/vgg16/"
    pre_trained_model_name = "vgg16"
    symbol_url = "http://data.mxnet.io/models/imagenet/vgg/vgg16-symbol.json"
    parameters_url = "http://data.mxnet.io/models/imagenet/vgg/vgg16-0000.params"
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

    # download pre-trained model
    url_download(symbol_url, dirname=pre_trained_path)
    url_download(parameters_url, dirname=pre_trained_path)
    ############################################################################

    ############################################################################
    # finetune model
    # wrap the input to fit the shape of the pre-trained model
    sym1 = mx.symbol.Variable('data')
    sym1 = mx.symbol.tile(sym1, reps=(1, 3, 1, 1))
    sym1 = mx.sym.UpSampling(sym1, scale=8, num_filter=1, name='up', sample_type='nearest')
    # sym1 = mx.symbol.FullyConnected(sym1, num_hidden=100)
    mod1 = mx.module.Module(
        sym1,
        data_names=['data', ],
        label_names=[],
        context=ctx
    )

    # load pre-trained model
    sym, arg_params, aux_params = mx.model.load_checkpoint(pre_trained_path + pre_trained_model_name, 0)

    label = mx.symbol.Variable('label')
    sym2, arg_params = get_fine_tune_model(sym, label, arg_params, 10, 'fc8')
    mod2 = mx.module.Module(
        sym2,
        data_names=['data', ],
        label_names=['label', ],
        context=ctx,
    )

    # link two mod
    mod = mx.module.SequentialModule(module_logger)

    mod.add(mod1)
    mod.add(mod2, take_labels=True, auto_wiring=True)

    # plot_network(
    #     nn_symbol=sym1,
    #     save_path=model_dir + "plot/network",
    #     shape={'data': (128, 3, 224, 224), 'label': (128,)},
    #     node_attrs={"fixedsize": "false"},
    #     show_tag=True
    # )
    #
    # plot_network(
    #     nn_symbol=sym2,
    #     save_path=model_dir + "plot/network",
    #     shape=form_shape(train_iter),
    #     node_attrs={"fixedsize": "false"},
    #     show_tag=True
    # )
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
        arg_params=arg_params,
        aux_params=aux_params,
    )

    ############################################################################
    # clean
    tins.close()
    ############################################################################