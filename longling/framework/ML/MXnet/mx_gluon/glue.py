# coding: utf-8
# created by tongshiwei on 18-2-3
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

import mxnet as mx
from mxnet import nd, autograd, gluon

from longling.lib.clock import Clock
from longling.lib.utilog import config_logging

from longling.framework.ML.MXnet.metric import PRF, Accuracy
from longling.framework.ML.MXnet.viz import plot_network
from longling.framework.ML.MXnet.mx_gluon.gluon_evaluater import ClassEvaluater
from longling.framework.ML.MXnet.mx_gluon.gluon_util import TrainBatchInfoer

from tqdm import tqdm
from collections import defaultdict
import json
import math

from longling.framework.ML.MXnet.mx_gluon.gluon_sym import PairwiseLoss


def build_map(filename):
    entities_map, e_idx = defaultdict(int), 1
    relations_map, r_idx = defaultdict(int), 1
    with open(filename) as f:
        for line in tqdm(f, desc="reading file[%s]" % filename):
            if not line.strip():
                continue
            pos_neg = json.loads(line)
            pos_sub, pos_rel, pos_obj = pos_neg["x"]
            neg_sub, neg_rel, neg_obj = pos_neg["z"]
            for entity in [pos_sub, pos_obj, neg_sub, neg_obj]:
                if entity not in entities_map:
                    entities_map[entity] = e_idx
                    e_idx += 1
            for relation in [pos_rel, neg_rel]:
                if relation not in relations_map:
                    relations_map[relation] = r_idx
                    r_idx += 1
    return entities_map, relations_map, e_idx, r_idx


class TransE(gluon.HybridBlock):
    def __init__(self,
                 entities_size, relations_size, dim=50,
                 **kwargs):
        super(TransE, self).__init__(**kwargs)

        with self.name_scope():
            self.entity_embedding = gluon.nn.Embedding(entities_size, dim,
                                                       weight_initializer=mx.init.Uniform(6 / math.sqrt(dim)))
            self.relation_embedding = gluon.nn.Embedding(relations_size, dim,
                                                         weight_initializer=mx.init.Uniform(6 / math.sqrt(dim)))

    def hybrid_forward(self, F, sub, rel, obj, **kwargs):
        sub = self.entity_embedding(sub)
        rel = self.relation_embedding(rel)
        obj = self.entity_embedding(obj)

        distance = F.add_n(sub, rel, mx.sym.negative(obj))
        return F.norm(distance, axis=1)


def transE():
    ############################################################################
    # parameters config
    # file path
    root = "../../../../"

    model_dir = root + "data/gluon/transE/"
    model_name = "transE"

    train_file = root + "data/KG/FB15/train.jsonxz"
    test_file = root + "data/KG/FB15/test.jsonxz"

    model_ctx = mx.cpu()

    batch_size = 128
    begin_epoch = 0
    epoch_num = 10

    # infoer
    bp_loss_f = {"pairwise_loss": PairwiseLoss(None, -1)}
    smoothing_constant = 0.01
    loss_function = {

    }
    loss_function.update(bp_loss_f)
    timer = Clock()
    batch_infoer = TrainBatchInfoer(loss_index=[name for name in loss_function], epoch_num=epoch_num - 1)

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

    def get_train_iter(filename):
        pos_subs, pos_rels, pos_objs = [], [], []
        neg_subs, neg_rels, neg_objs = [], [], []
        with open(filename) as f:
            for line in tqdm(f, desc="reading file[%s]" % filename):
                if not line.strip():
                    continue
                pos_neg = json.loads(line)
                pos_sub, pos_rel, pos_obj = pos_neg["x"]
                pos_subs.append(entities_map[pos_sub])
                pos_rels.append(relations_map[pos_rel])
                pos_objs.append(entities_map[pos_obj])
                neg_sub, neg_rel, neg_obj = pos_neg["z"]
                neg_subs.append(entities_map[neg_sub])
                neg_rels.append(relations_map[neg_rel])
                neg_objs.append(entities_map[neg_obj])

        return gluon.data.DataLoader(
            gluon.data.ArrayDataset(pos_subs, pos_rels, pos_objs, neg_subs, neg_rels, neg_objs), batch_size=batch_size,
            shuffle=True)

    def get_test_iter(filename):
        pos_negs = []
        with open(filename) as f:
            for i, line in tqdm(enumerate(f), desc="reading file[%s]" % filename):
                if not line.strip():
                    continue
                pos_neg = json.loads(line)
                pos_sub, pos_rel, pos_obj = pos_neg["x"]
                pos_sub = entities_map[pos_sub]
                pos_rel = relations_map[pos_rel]
                pos_obj = entities_map[pos_obj]
                negs = []
                for neg_triple in pos_neg["z"]:
                    neg_sub, neg_rel, neg_obj = neg_triple
                    negs.append((entities_map[neg_sub], relations_map[neg_rel], entities_map[neg_obj]))
                pos_negs.append([(pos_sub, pos_rel, pos_obj)] + negs)
                if i > 10:
                    break
        return pos_negs

    train_data = get_train_iter(train_file)
    test_data = get_test_iter(test_file)

    ############################################################################
    # network building
    net = TransE(
        entities_size=entities_size,
        relations_size=relations_size,
        dim=50,
    )
    net.hybridize()

    ############################################################################
    # visulization
    sub = mx.sym.var("sub")
    rel = mx.sym.var("rel")
    obj = mx.sym.var("obj")
    sym = net(sub, rel, obj)
    plot_network(
        nn_symbol=sym,
        save_path=model_dir + "plot/network",
        shape=viz_shape,
        node_attrs={"fixedsize": "false"},
        view=False
    )
    ############################################################################

    # epoch training
    net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})

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
            #
            if i % 1 == 0:
                loss_values = [loss for loss in moving_losses.values()]
                batch_infoer.report(i, loss_value=loss_values)
        batch_infoer.batch_end(i)

        train_time = timer.end(wall=True)

        for data in tqdm(test_data, "testing"):
            subs, rels, objs = [], [], []
            for d in data:
                subs.append(d[0])
                rels.append(d[1])
                objs.append(d[2])
            eval_data = gluon.data.ArrayDataset(subs, rels, objs),
            res = net(eval_data)
            print(res)

    #     test_eval_res = evaluater.evaluate(test_data, net, stage="test")
    #     print(evaluater.format_eval_res(epoch, test_eval_res, loss_values, train_time,
    #                                     logger=evaluater.logger, log_f=evaluater.log_f)[0])
    #     net.save_params(model_dir + model_name + "-%04d.parmas" % (epoch + 1))
    # net.export(model_dir + model_name)
    #
    # ############################################################################
    # # clean
    # evaluater.log_f.close()
    # ############################################################################


if __name__ == '__main__':
    transE()
