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
from longling.framework.ML.MXnet.mx_gluon.gluon_evaluater import Evaluater
from longling.framework.ML.MXnet.mx_gluon.gluon_util import TrainBatchInfoer

from tqdm import tqdm
from collections import defaultdict
import json
import math

from longling.framework.ML.MXnet.mx_gluon.gluon_sym import PairwiseLoss
from longling.lib.stream import wf_open


def eval(test_data_part, net, model_ctx, top_n=10):
    topn = 0
    for i, data in tqdm(enumerate(test_data_part), "testing"):
        subs, rels, objs = [], [], []
        for d in data:
            subs.append(d[0])
            rels.append(d[1])
            objs.append(d[2])
        eval_data = gluon.data.DataLoader(gluon.data.ArrayDataset(subs, rels, objs),
                                          batch_size=len(subs), shuffle=False)

        res = None
        for (sub, rel, obj) in eval_data:
            sub = sub.as_in_context(model_ctx)
            rel = sub.as_in_context(model_ctx)
            obj = sub.as_in_context(model_ctx)
            if res is None:
                res = net(sub, rel, obj)
            else:
                res.concat(net(sub, rel, obj))
        res = res.asnumpy().tolist()
        smaller = 0
        topn += 1
        for n in res[1:]:
            if n < res[0]:
                smaller += 1
            if smaller >= top_n:
                topn -= 1
                break
    return topn


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


def get_l2_embedding_weight(F, embedding_size, batch_size=None, prefix=""):
    entries = list(range(embedding_size))
    embedding_weight = []
    batch_size = batch_size if batch_size else embedding_size
    for entity in tqdm(gluon.data.DataLoader(gluon.data.ArrayDataset(entries),
                                             batch_size=batch_size, shuffle=False),
                       'getting %s embedding' % prefix):
        embedding_weight.extend(mx.nd.L2Normalization(F(entity)).asnumpy().tolist())

    return embedding_weight


def embedding2file(filename, embedding_weight, embedding_map):
    with wf_open(filename) as wf:
        for thing_id, idx in tqdm(embedding_map.items(), filename):
            print("%s %s" % (thing_id,
                             " ".join([str(float('%.6f' % embedding)) for embedding in embedding_weight[idx]])),
                  file=wf)


class TransE(gluon.HybridBlock):
    def __init__(self,
                 entities_size, relations_size, dim=50,
                 d='L1', **kwargs):
        super(TransE, self).__init__(**kwargs)
        self.d = d

        with self.name_scope():
            self.entity_embedding = gluon.nn.Embedding(entities_size, dim,
                                                       weight_initializer=mx.init.Uniform(6 / math.sqrt(dim)))

            self.relation_embedding = gluon.nn.Embedding(relations_size, dim,
                                                         weight_initializer=mx.init.Uniform(6 / math.sqrt(dim)))

            self.batch_norm = gluon.nn.BatchNorm()

    def hybrid_forward(self, F, sub, rel, obj, **kwargs):
        sub = self.entity_embedding(sub)
        rel = self.relation_embedding(rel)
        obj = self.entity_embedding(obj)

        sub = F.L2Normalization(sub)
        rel = F.L2Normalization(rel)
        obj = F.L2Normalization(obj)

        sub = self.batch_norm(sub)
        rel = self.batch_norm(rel)
        obj = self.batch_norm(obj)
        distance = F.add_n(sub, rel, F.negative(obj))
        if self.d == 'L2':
            return F.norm(distance, axis=1)
        elif self.d == 'L1':
            return F.sum(F.abs(distance), axis=1)


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
    evaluater = Evaluater(
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

    print("entities_size: %s | relations_size: %s" % (entities_size, relations_size))

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
        with open(filename) as f:
            for i, line in f:
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
                yield [(pos_sub, pos_rel, pos_obj)]

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
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})

    def embedding_persistence():
        entity_embedding = get_l2_embedding_weight(net.entity_embedding, entities_size, batch_size=batch_size,
                                                   prefix="entity")
        embedding2file(vec_dir + "entity.vec.dat", entity_embedding, entities_map)
        relation_embedding = get_l2_embedding_weight(net.relation_embedding, relations_size, batch_size=batch_size,
                                                   prefix="relation")
        embedding2file(vec_dir + "relation.vec.dat", relation_embedding, relations_map)

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
                batch_infoer.report(i, loss_value=loss_values)
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


if __name__ == '__main__':
    transE()
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
