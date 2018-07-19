# coding: utf-8
# created by tongshiwei on 18-2-3
from __future__ import absolute_import
from __future__ import print_function

import json
import math
import os
import sys
from collections import defaultdict

import re
import mxnet as mx
from mxnet import nd, autograd, gluon
from tqdm import tqdm

from longling.framework.ML.MXnet.mx_gluon.gluon_sym import PairwiseLoss
from longling.framework.ML.MXnet.mx_gluon.gluon_toolkit.evaluator import Evaluator
from longling.framework.ML.MXnet.mx_gluon.gluon_toolkit.informer import TrainBatchInformer
from longling.framework.ML.MXnet.viz import plot_network
from longling.lib.clock import Clock
from longling.lib.stream import wf_open
from longling.lib.utilog import config_logging, LogLevel



def transE():
    ############################################################################
    # parameters config
    dim = 50
    gamma = 1
    d = 'L1'
    lr = 0.01
    processer = 1
    from multiprocessing import Pool
    top_n = 10

    # file path
    root = "../../../../"

    model_dir = root + "data/gluon/transE/"
    model_name = "transE"

    train_file = root + "data/KG/FB15/train.jsonxz"
    test_file = root + "data/KG/FB15/test.jsonxz"
    vec_dir = root + "data/KG/FB15/"
    validation_result_file = model_dir + "result"

    model_ctx = mx.cpu()

    batch_size = 128
    begin_epoch = 0
    epoch_num = 200

    # infoer
    validation_logger = config_logging(
        filename=model_dir + "result.log",
        logger="validation",
        mode="w",
        log_format="%(message)s",
    )
    evaluater = Evaluator(
        # metrics=eval_metrics,
        model_ctx=model_ctx,
        logger=validation_logger,
        log_f=validation_result_file
    )

    bp_loss_f = {"pairwise_loss": PairwiseLoss(None, -1, margin=gamma)}
    smoothing_constant = 0.01
    loss_function = {

    }
    loss_function.update(bp_loss_f)
    timer = Clock()
    batch_infoer = TrainBatchInformer(loss_index=[name for name in loss_function], epoch_num=epoch_num - 1)

    # viz var
    data_shape = (1,)
    viz_shape = {
        'sub': (batch_size,) + data_shape,
        'rel': (batch_size,) + data_shape,
        'obj': (batch_size,) + data_shape,
    }
    ############################################################################

    ############################################################################
    entities_map, relations_map, entities_size, relations_size = build_map(train_file)


    train_data = get_train_iter(train_file)
    test_data = get_test_iter(test_file)
    if processer > 1:
        p_size = math.floor(len(test_data) / processer)
        test_data_parts = [test_data[k * p_size: (k + 1) * p_size + 1] for k in range(processer)]
    else:
        test_data_parts = None

    ############################################################################
    # network building
    net = TransE(
        entities_size=entities_size,
        relations_size=relations_size,
        dim=dim,
        d=d,
    )
    net.hybridize()

    # epoch training
    net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})



    for epoch in range(begin_epoch, epoch_num):
        # initial
        timer.start()
        batch_infoer.batch_start(epoch)
        moving_losses = {name: 0 for name in loss_function}
        bp_loss = None

        # batch training
        for i, (pos_sub, pos_rel, pos_obj, neg_sub, neg_rel, neg_obj) in enumerate(train_data):
            pos_sub = pos_sub.as_in_context(model_ctx)
            pos_rel = pos_rel.as_in_context(model_ctx)
            pos_obj = pos_obj.as_in_context(model_ctx)
            neg_sub = neg_sub.as_in_context(model_ctx)
            neg_rel = neg_rel.as_in_context(model_ctx)
            neg_obj = neg_obj.as_in_context(model_ctx)
            with autograd.record():
                neg_out = net(pos_sub, pos_rel, pos_obj)
                pos_out = net(neg_sub, neg_rel, neg_obj)
                for name, function in loss_function.items():
                    loss = function(pos_out, neg_out)
                    if name in bp_loss_f:
                        bp_loss = loss
                    loss_value = nd.mean(loss).asscalar()
                    moving_losses[name] = (loss_value if ((i == 0) and (epoch == 0))
                                           else (1 - smoothing_constant) * moving_losses[
                        name] + smoothing_constant * loss_value)
            # assert bp_loss is not None
            assert bp_loss is not None
            bp_loss.backward()
            trainer.step(batch_size=batch_size)

            if i % 1 == 0:
                loss_values = [loss for loss in moving_losses.values()]
                batch_infoer.batch_report(i, loss_value=loss_values)
        batch_infoer.batch_end(i)

        train_time = timer.end(wall=True)

        if processer > 1:
            pool = Pool()
            topns = []
            for i, test_data_part in enumerate(test_data_parts):
                topns.append(pool.apply_async(eval, args=(test_data_part, net, model_ctx, top_n)))
            pool.close()
            pool.join()
            topn_res = sum([topn.get() for topn in topns]) / len(test_data)
        else:
            topn_res = eval(test_data, net, model_ctx, top_n) / len(test_data)

        loss_values = {name: loss for name, loss in moving_losses.items()}.items()
        print(evaluater.format_eval_res(epoch, {'hits@%s' % top_n: topn_res}, loss_values, train_time,
                                        logger=evaluater.logger, log_f=evaluater.log_f)[0])
        net.save_params(model_dir + model_name + "-%04d.parmas" % (epoch + 1))

        if i % 20 == 0:
            embedding_persistence()
    embedding_persistence()
    net.export(model_dir + model_name)

    ############################################################################
    # clean
    evaluater.log_f.close()
    ############################################################################

logger = config_logging(logger="glue", console_log_level=LogLevel.INFO)


def new_module(module_name, directory=None):
    glum_directory = os.path.dirname(sys._getframe().f_code.co_filename)
    glum_py = os.path.join(glum_directory, "glum.py")
    module_filename = module_name + ".py"
    target = os.path.join(directory, module_filename) if directory else module_filename
    if os.path.isfile(target):
        logger.error("file already existed, will not override, generation abort")
        return False
    logger.info("generating file, path is %s", target)
    big_module_name = "%sModule" % (module_name[0].upper() + module_name[1:])
    with open(glum_py, encoding="utf-8") as f, wf_open(target) as wf:
        for line in f:
            print(line.replace("module_name", module_name).replace("GluonModule", big_module_name), end="", file=wf)
    return True

if __name__ == '__main__':
    new_module("RLSTM")
    # transE()
    # net = TransE(
    #     entities_size=100,
    #     relations_size=100,
    #     dim=5,
    #     d='L1',
    # )
    # net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=mx.cpu())
    # net(nd.array([1, 1, 1]), nd.array([2, 3, 4]), nd.array([2, 2, 2]))

    # a = [
    #     [1, 2, 3, 4, 5],
    #     [2, 2, 2, 3, 3],
    #     [3, 4, 4, 5, 5],
    # ]
    # a = mx.nd.array(a)
    # for d in mx.nd.L2Normalization(a).asnumpy():
    #     print(d.tolist())
    # # print(a / mx.nd.norm(a * a, axis=-1, keepdims=True))
