# coding: utf-8
# created by tongshiwei on 18-1-27


from __future__ import absolute_import

import json
import logging

import mxnet as mx
import numpy as np

from tqdm import tqdm

from longling.framework.ML.mxnet.callback import tqdm_speedometer, do_checkpoint, \
    ClassificationLogValidationMetricsCallback, Speedometer, \
    TqdmEpochReset
from longling.framework.ML.mxnet.metric import PRF, Accuracy, CrossEntropy
from longling.framework.ML.mxnet.viz import plot_network, form_shape
from longling.framework.ML.mxnet.io_lib import DictJsonIter, VecDict, SimpleBucketIter
from longling.framework.ML.mxnet.monitor import TimeMonitor
from longling.framework.ML.mxnet.sym_lib import dnn_cell, lenet_cell, text_cnn_cell
from longling.framework.ML.mxnet.util import get_fine_tune_model, BasePredictor, RNNPredictor

from longling.lib.utilog import config_logging
from longling.framework.spider import url_download


def dnn():
    ############################################################################
    # parameters config

    # file path
    root = "../../../../"
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
        format="%(message)s",
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
        show_tag=True
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


def cnn():
    ############################################################################
    # parameters config

    # file path
    root = "../../../../"
    train_file = root + "data/image/mnist_train"
    test_file = root + "data/image/mnist_test"

    model_dir = root + "data/mxnet/cnn/"
    model_name = "cnn"

    # model args
    batch_size = 128
    last_batch_handle = 'pad'

    num_label = 10

    begin_epoch = 0
    num_epoch = 20
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
        format="%(message)s",
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
    def sym_gen(num_label):
        data = mx.symbol.Variable('data')
        label = mx.symbol.Variable('label')
        data = lenet_cell(
            in_sym=data,
            num_output=num_label,
        )
        sym = mx.symbol.SoftmaxOutput(
            data=data,
            label=label,
        )
        return sym

    sym = sym_gen(num_label)
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
        show_tag=True
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


def cnn_use():
    root = "../../../../"
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
        im0 = Image.fromarray(np.uint8(data.asnumpy()[0][0]*256), mode='L')
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

def text_cnn():
    ############################################################################
    # parameters config
    logging.getLogger().setLevel(logging.INFO)
    propagate = False

    root = "../../../../"

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
        format="%(message)s",
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
        show_tag=True
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


def vgg16():
    ############################################################################
    # parameters config
    # logging.getLogger().setLevel(logging.INFO)
    propagate = False

    root = "../../../../"
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


def text_rnn():
    ############################################################################
    # parameters config
    # file path
    root = "../../../../"

    dict_file = root + "data/word2vec/comment.vec.dat"
    train_file = root + "data/text/mini.instance.train"
    test_file = root + "data/text/mini.instance.test"

    model_dir = root + "data/mxnet/text_rnn/"
    model_name = "text_rnn"

    # model args
    batch_size = 128

    begin_epoch = 0
    num_epoch = 5
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
        filename=model_dir + "result.log",
        logger="validation",
        mode="w",
        format="%(message)s",
        propagate=propagate,
    )
    validation_result_file = model_dir + "result"

    buckets = [10, 20, 30, 40, 50, 60]
    num_layers = 2
    num_hidden = 100
    num_label = 2
    dropout = 0.5

    # viz arg
    viz_seq_len = 5

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
                datas.extend(w2v_dict.s2i([data['x']], sentence_split=True).tolist())
                labels.append(int(data['z']))

        datas_labels = zip(datas, labels)
        datas_labels = SimpleBucketIter.bucket_sort(datas_labels, sorted_key=lambda x: len(x[0]))

        datas_labels, _, _ = datas_labels
        datas, labels = zip(*datas_labels)

        return SimpleBucketIter(
            data=datas,
            label=labels,
            batch_size=batch_size,
            buckets=buckets,
            data_name='data',
            label_name='label',
            label_shape=1,
            padding_num=0,
        )

    train_data = get_iter(train_file)
    test_data = get_iter(test_file)

    ############################################################################

    ############################################################################
    # network building
    # stack = mx.rnn.SequentialRNNCell()
    # for i in range(num_layers):
    #     stack.add(mx.rnn.LSTMCell(num_hidden=num_hidden, prefix='lstm_cell%d_' % i))
    #     gru_cell = mx.rnn.GRUCell(num_hidden=num_hidden, prefix="gru_cell%d_" % i)
    # stack.add(mx.rnn.DropoutCell(dropout=dropout))
    # cells = stack

    i = 0
    # bi_cell_params = mx.rnn.RNNParams("lstm_cell%d_" % i)
    lstm_cell = mx.rnn.LSTMCell(
        num_hidden=num_hidden,
        prefix='lstm_cell%d_' % i,
        # params=bi_cell_params,
    )
    # lstm_cell = mx.rnn.LSTMCell(
    #     num_hidden=num_hidden,
    #     prefix='lstm_cell%d_' % i,
    #     params=bi_cell_params,
    # )
    stack = mx.rnn.BidirectionalCell(
        lstm_cell,
        lstm_cell,
    )
    cells = [lstm_cell]

    # pay attention to bi-lstm when two lstm shared the same variable
    # manually do save and load and check them carefully

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('label')

        embed = mx.sym.Embedding(data=data, input_dim=vocab_size,
                                 output_dim=vec_size, name='word_embedding')

        # embed = mx.symbol.BlockGrad(embed)
        # not support cpu yet
        # embed = mx.symbol.RNN(data=embed, mode="lstm", num_layers=seq_len, state_size=100, state_outputs=False)
        # pred = mx.symbol.RNN(data=embed, mode="lstm", num_layers=seq_len, state_size=100, state_outputs=False)

        # stack.reset()
        # concat outputs
        outputs, states = stack.unroll(seq_len, inputs=embed, merge_outputs=True)
        pred = mx.symbol.mean(outputs, axis=1)
        # pred = mx.symbol.max(outputs, axis=1)
        # pred = mx.symbol.min(outputs, axis=1)

        # multi outputs
        # outputs, states = stack.unroll(seq_len, inputs=embed, merge_outputs=False)
        # pred = outputs[-1]
        # pred = outputs[0] * np.exp(1 - len(outputs))
        # for idx, output in enumerate(outputs[1:]):
        #     pred_sub = output * np.exp((2 + idx) - len(outputs))
        #     pred = pred + pred_sub

        pred = mx.sym.FullyConnected(data=pred, num_hidden=num_label, name='pred2')

        # label = mx.sym.Reshape(label, shape=(-1,))
        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return pred, ('data',), ('label',)

    mod = mx.mod.BucketingModule(
        sym_gen=sym_gen,
        default_bucket_key=train_data.default_bucket_key,
        context=ctx,
        logger=module_logger,
    )

    sym = sym_gen(viz_seq_len)[0]
    plot_network(
        nn_symbol=sym,
        save_path=model_dir + "plot/network",
        shape={'data': (batch_size, viz_seq_len), 'label': (batch_size,)},
        node_attrs={"fixedsize": "false"},
        show_tag=True,
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
            loss_metrics=[cross_entropy],
            logger=validation_logger,
            logfile=validation_result_file,
        ),
        # kvstore="devices",
        # eval_batch_end_callback=[vd_logger1],
        begin_epoch=begin_epoch,
        num_epoch=num_epoch,
        optimizer='rmsprop',
        arg_params={'word_embedding_weight': mx.nd.array(embedding)},
        allow_missing=True,
        optimizer_params={'learning_rate': 0.0005},
        batch_end_callback=[speedometer, tins],
        epoch_end_callback=[
            mx.rnn.do_rnn_checkpoint(cells, model_dir + model_name),
            TqdmEpochReset(tins, "training"),
        ],
        monitor=time_monitor,
    )
    ###########################################################################

    ###########################################################################
    # clean
    tins.close()


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
    # dnn()
    # text_cnn()
    # vgg16()
    # text_rnn()
    # rnn()
    # cnn()
    cnn_use()

