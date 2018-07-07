# coding:utf-8
# created by tongshiwei on 2018/7/6

from __future__ import absolute_import

import json
import logging

import mxnet as mx

from tqdm import tqdm

from longling.framework.ML.MXnet.callback import tqdm_speedometer, \
    ClassificationLogValidationMetricsCallback, Speedometer, \
    TqdmEpochReset
from longling.framework.ML.MXnet.metric import PRF, Accuracy, CrossEntropy
from longling.framework.ML.MXnet.viz import plot_network, form_shape
from longling.framework.ML.MXnet.io_lib import DictJsonIter, VecDict, SimpleBucketIter
from longling.framework.ML.MXnet.monitor import TimeMonitor

from longling.lib.utilog import config_logging


def text_rnn():
    ############################################################################
    # parameters config
    # file path
    root = "../../../../../"

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
        log_format="%(message)s",
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
        view=True,
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
