# coding:utf-8
# created by tongshiwei on 2018/7/11

from __future__ import absolute_import
from __future__ import print_function


import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon

from longling.lib.clock import Clock
from longling.lib.utilog import config_logging

from longling.framework.ML.MXnet.metric import PRF, Accuracy
from longling.framework.ML.MXnet.viz import plot_network
from longling.framework.ML.MXnet.mx_gluon.gluon_evaluater import ClassEvaluater
from longling.framework.ML.MXnet.mx_gluon.gluon_util import TrainBatchInfoer



def cnn():
    ############################################################################
    # parameters config
    # file path
    root = "../../../../"

    model_dir = root + "data/gluon/cnn/"
    model_name = "cnn"

    data_ctx = mx.cpu()

    def transform(data, label):
        return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)

    model_ctx = mx.cpu()

    num_outputs = 10
    num_hiddens = [512]

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
    batch_infoer = TrainBatchInfoer(loss_index=[name for name in loss_function], epoch_num=epoch_num - 1)
    evaluater = ClassEvaluater(
        metrics=eval_metrics,
        model_ctx=model_ctx,
        logger=validation_logger,
        log_f=validation_result_file
    )
    # viz var
    data_shape = (1, 32, 32)
    viz_shape = {'data': (batch_size,) + data_shape}
    ############################################################################

    ############################################################################
    # network building
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # The Flatten layer collapses all axis, except the first one, into one axis.
        net.add(gluon.nn.Flatten())
        for num_hidden in num_hiddens:
            net.add(gluon.nn.Dense(num_hidden, activation="relu"))
        net.add(gluon.nn.Dense(num_outputs))
    net.hybridize()

    ############################################################################
    # visulization
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
    # loading data
    train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                          batch_size, shuffle=True)
    test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                         batch_size, shuffle=False)
    ############################################################################
    # epoch training
    net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)
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
                batch_infoer.report(i, loss_value=loss_values)

        if 'num_inst' not in locals().keys() or num_inst is None:
            num_inst = (i + 1) * batch_size
            assert num_inst is not None

        loss_values = {name: loss for name, loss in moving_losses.items()}.items()
        batch_infoer.batch_end(i)
        train_time = timer.end(wall=True)
        test_eval_res = evaluater.evaluate(test_data, net, stage="test")
        print(evaluater.format_eval_res(epoch, test_eval_res, loss_values, train_time,
                                        logger=evaluater.logger, log_f=evaluater.log_f)[0])
        net.save_params(model_dir + model_name + "-%04d.parmas" % (epoch + 1))
    net.export(model_dir + model_name)

    ############################################################################
    # clean
    evaluater.log_f.close()
    ############################################################################