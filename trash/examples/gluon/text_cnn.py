# coding:utf-8
# created by tongshiwei on 2018/7/11
from __future__ import absolute_import
from __future__ import print_function

import json

import mxnet as mx
from mxnet import nd, autograd, gluon
from tqdm import tqdm

from trash.io import VecDict
from longling.framework.ML.MXnet.metric import PRF, Accuracy
from longling.ML.MxnetHelper.mx_gluon import TextCNN
from longling.ML.MxnetHelper.mx_gluon import ClassEvaluator
from longling.ML.MxnetHelper.mx_gluon import TrainBatchInformer
from longling.framework.ML.MXnet.viz import plot_network
from longling.lib.clock import Clock
from longling.lib.utilog import config_logging


def text_cnn():
    ############################################################################
    # parameters config
    # file path
    root = "../../../../"

    model_dir = root + "data/gluon/text_cnn/"
    model_name = "text_cnn"

    dict_file = root + "data/word2vec/comment.vec.dat"
    train_file = root + "data/text/mini.instance.train"
    test_file = root + "data/text/mini.instance.test"

    data_ctx = mx.cpu()
    model_ctx = mx.cpu()

    sentence_size = 25
    num_outputs = 2

    batch_size = 128
    begin_epoch = 0
    epoch_num = 10
    bp_loss_f = {"cross-entropy": gluon.loss.SoftmaxCrossEntropyLoss()}
    smoothing_constant = 0.01
    loss_function = {

    }
    loss_function.update(bp_loss_f)

    # infoer
    propagate = False
    validation_logger = config_logging(
        filename=model_dir + "result.log",
        logger="validation",
        mode="w",
        log_format="%(message)s",
        propagate=propagate,
    )
    validation_result_file = model_dir + "result"

    timer = Clock()
    eval_metrics = [PRF(argmax=False), Accuracy(argmax=False)]
    batch_infoer = TrainBatchInformer(loss_index=[name for name in loss_function], end_epoch=epoch_num - 1)
    evaluater = ClassEvaluator(
        metrics=eval_metrics,
        model_ctx=model_ctx,
        logger=validation_logger,
        log_f=validation_result_file
    )
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

        return gluon.data.DataLoader(gluon.data.ArrayDataset(datas, labels), batch_size=batch_size, shuffle=True)

    train_data = get_iter(train_file)
    test_data = get_iter(test_file)
    ############################################################################

    ############################################################################
    # network building
    net = TextCNN(sentence_size, vec_size, vocab_size=vocab_size, dropout=0.5, batch_norm=1, num_highway=True)
    net.hybridize()
    ############################################################################

    ############################################################################
    # visulization
    # viz var
    data_shape = (sentence_size,)
    viz_shape = {'data': (batch_size,) + data_shape}

    x = mx.sym.var("data")
    sym = net(x)
    plot_network(
        nn_symbol=sym,
        save_path=model_dir + "plot/network",
        shape=viz_shape,
        node_attrs={"fixedsize": "false"},
        view=False
    )
    ############################################################################

    ############################################################################
    # epoch training
    net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)
    net.embedding.weight.set_data(nd.array(embedding).as_in_context(model_ctx))
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})

    for epoch in range(begin_epoch, epoch_num):
        # initial
        moving_losses = {name: 0 for name in loss_function}
        batch_infoer.batch_start(epoch)
        timer.start()

        # batch training
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx)
            label = label.as_in_context(model_ctx)
            bp_loss = None
            with autograd.record():
                output = net(data)
                for name, function in loss_function.items():
                    loss = function(output, label)
                    if name in bp_loss_f:
                        bp_loss = loss
                    loss_value = nd.mean(loss).asscalar()
                    moving_losses[name] = (loss_value if ((i == 0) and (epoch == 0))
                                           else (1 - smoothing_constant) * moving_losses[
                        name] + smoothing_constant * loss_value)

            assert bp_loss is not None
            bp_loss.backward()
            trainer.step(data.shape[0])

            if i % 1 == 0:
                loss_values = [loss for loss in moving_losses.values()]
                batch_infoer.__call__(i, loss_value=loss_values)

        if 'num_inst' not in locals().keys() or num_inst is None:
            num_inst = (i + 1) * batch_size
            assert num_inst is not None

        loss_values = {name: loss for name, loss in moving_losses.items()}.items()
        batch_infoer.batch_end(i)
        train_time = timer.end(wall=True)
        test_eval_res = evaluater.evaluate(test_data, net, stage="test")
        print(evaluater.__call__(epoch, test_eval_res, loss_values, train_time,
                                 logger=evaluater.logger, log_f=evaluater.log_f)[0])
        net.save_params(model_dir + model_name + "-%04d.parmas" % (epoch + 1))
    net.export(model_dir + model_name)

    ############################################################################
    # clean
    evaluater.log_f.close()
    ############################################################################