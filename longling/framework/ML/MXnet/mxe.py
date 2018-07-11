# coding: utf-8
# created by tongshiwei on 18-1-27


from __future__ import absolute_import
from __future__ import division

import json
import logging
import math
import time

from collections import defaultdict

import mxnet as mx
import numpy as np

from tqdm import tqdm

from longling.framework.ML.MXnet.callback import tqdm_speedometer, \
    ClassificationLogValidationMetricsCallback, Speedometer, \
    TqdmEpochReset
from longling.framework.ML.MXnet.metric import PRF, Accuracy, CrossEntropy, PairwiseMetric
from longling.framework.ML.MXnet.viz import plot_network, form_shape
from longling.framework.ML.MXnet.io_lib import DictJsonIter, VecDict, SimpleBucketIter
from longling.framework.ML.MXnet.monitor import TimeMonitor
from longling.framework.ML.MXnet.sym_lib import lenet_cell, text_cnn_cell, mx_constant
from longling.framework.ML.MXnet.util import get_fine_tune_model
from longling.framework.ML.MXnet.sym_lib import pairwise_loss

from longling.lib.utilog import config_logging
from longling.lib.candylib import as_list
_as_list = as_list

def text_sim_cnn():
    vocab_size = 365000
    vec_size = 80
    sentence_size = 25
    root = "../../../"
    model_dir = root + "data/mxnet/text_cnn/"
    model_name = "text_cnn"

    sentence_embedding_size = 200

    ############################################################################
    # network building

    def sym_gen(filter_list, num_filter, vocab_size, vec_size, sentence_size, num_label):
        data1 = mx.symbol.Variable('data1')
        data2 = mx.symbol.Variable('data2')
        embed_weight = mx.symbol.Variable('word_embedding')

        def sub_sym_gen(data):
            data = mx.symbol.expand_dims(data, axis=1)
            data = mx.symbol.Embedding(data, weight=embed_weight, input_dim=vocab_size,
                                       output_dim=vec_size, name="word_embedding")
            data = mx.sym.BlockGrad(data)
            data = text_cnn_cell(
                in_sym=data,
                sentence_size=sentence_size,
                vec_size=80,
                num_output=sentence_embedding_size,
                filter_list=filter_list,
                num_filter=num_filter,
                dropout=0.5,
                batch_norms=1,
                highway=True,
                name_prefix=model_name,
            )

            data = mx.sym.softmax(data)
            return data

        data1 = sub_sym_gen(data1)
        data2 = sub_sym_gen(data2)

        def pairwise_loss(pos_sym, neg_sym, margin):
            pos_sym = mx.sym.slice_axis(pos_sym, axis=1, begin=0, end=1)
            neg_sym = mx.sym.slice_axis(neg_sym, axis=1, begin=0, end=1)
            margin = mx_constant(margin)
            loss = mx.sym.add_n(mx.sym.negative(pos_sym), neg_sym, margin)
            sym = mx.sym.relu(loss)
            return sym

        sym = pairwise_loss(data1, data2, margin=0.2)
        return sym

    sym = sym_gen(
        filter_list=[1, 2, 3, 4],
        num_filter=60,
        vocab_size=vocab_size,
        vec_size=vec_size,
        sentence_size=sentence_size,
        num_label=2,
    )

    plot_network(
        nn_symbol=sym,
        save_path=model_dir + "plot/network",
        node_attrs={"fixedsize": "false"},
        shape={
            'data1': (10, 25),
            'data2': (10, 25),
        },
        view=True
    )
    ############################################################################


# todo
# plot data on website
# implement Jin experiment(r-cnn)
# GAN
# poem rnn
# picture to poem
# rcnn review
# rl network
# mxnet imgae iter

def transE_cell(sub, rel, obj,
                sub_embed, sub_embed_size,
                rel_embed, rel_embed_size,
                obj_embed=None, obj_embed_size=None,
                sub_embed_name=None, obj_embed_name=None, rel_embed_name=None,
                output_dim=50):
    if obj_embed is None:
        obj_embed = sub_embed
        obj_embed_size = sub_embed_size
        obj_embed_name = sub_embed_name

    if sub_embed_name is None:
        sub_embed_name = sub_embed.name
    if rel_embed_name is None:
        rel_embed_name = rel_embed.name
    if obj_embed_name is None:
        obj_embed_name = obj_embed.name

    sub = mx.sym.Embedding(sub, weight=sub_embed, input_dim=sub_embed_size, output_dim=output_dim,
                           name=sub_embed_name)
    obj = mx.sym.Embedding(obj, weight=obj_embed, input_dim=obj_embed_size, output_dim=output_dim,
                           name=obj_embed_name)
    rel = mx.sym.Embedding(rel, weight=rel_embed, input_dim=rel_embed_size, output_dim=output_dim,
                           name=rel_embed_name)

    distance = mx.sym.add_n(sub, rel, mx.sym.negative(obj))
    sym = mx.sym.norm(distance, axis=1)
    return sym


def transE():
    ############################################################################
    # parameters config
    margin = 0.1
    dim = 60

    # file path
    root = "../../../"
    train_file = root + "data/KG/FB15/train.jsonxz"
    test_file = root + "data/KG/FB15/test.jsonxz"

    model_dir = root + "data/mxnet/transE/"
    model_name = "transE"

    # model args
    batch_size = 128
    last_batch_handle = 'pad'

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
        log_format="%(message)s",
        propagate=propagate,
    )
    validation_result_file = model_dir + "result"

    ############################################################################

    ############################################################################
    # loading data and build dict
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

    entities_map, relations_map, e_input_dim, r_input_dim = build_map(train_file)

    class NDArrayIter(mx.io.NDArrayIter):
        @property
        def provide_label(self):
            return None

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

        return NDArrayIter(
            data={
                'pos_sub': np.asarray(pos_subs),
                'pos_rel': np.asarray(pos_rels),
                'pos_obj': np.asarray(pos_objs),
                'neg_sub': np.asarray(neg_subs),
                'neg_rel': np.asarray(neg_rels),
                'neg_obj': np.asarray(neg_objs),
            },
            batch_size=batch_size, last_batch_handle=last_batch_handle,
        )

    def get_test_iter(filename):
        subs, rels, objs = [], [], []
        labels = []
        with open(filename) as f:
            for line in tqdm(f, desc="reading file[%s]" % filename):
                if not line.strip():
                    continue
                data = json.loads(line)
                sub, rel, obj = data['x']
                label = data['z']
                subs.append(entities_map[sub])
                rels.append(relations_map[rel])
                objs.append(entities_map[obj])
                labels.append(label)

        return NDArrayIter(
            data={
                'sub': np.asarray(subs),
                'rel': np.asarray(rels),
                'obj': np.asarray(objs),
            },
            # label={'label': np.array(labels)},
            label=None,
            batch_size=batch_size, last_batch_handle=last_batch_handle,
        )

    train_data = get_train_iter(train_file)
    test_data = get_test_iter(test_file)

    ############################################################################

    ############################################################################
    # network building
    entity_embed = mx.sym.Variable("EntityEmbedding", init=mx.init.Uniform(6 / math.sqrt(dim)))
    relation_embed = mx.sym.Variable("RelationEmbedding", init=mx.init.Uniform(6 / math.sqrt(dim)))

    def transE_body(sub, rel, obj):
        return transE_cell(
            sub, obj, rel,
            sub_embed=entity_embed, sub_embed_size=e_input_dim,
            rel_embed=relation_embed, rel_embed_size=r_input_dim,
            output_dim=dim,
        )

    def pairwise_sym_gen(margin):
        pos_sub = mx.sym.Variable('pos_sub')
        pos_obj = mx.sym.Variable('pos_obj')
        pos_rel = mx.sym.Variable('pos_rel')

        neg_sub = mx.sym.Variable('neg_sub')
        neg_obj = mx.sym.Variable('neg_obj')
        neg_rel = mx.sym.Variable('neg_rel')

        pos_sym = transE_body(pos_sub, pos_rel, pos_obj)
        neg_sym = transE_body(neg_sub, neg_obj, neg_rel)

        return pairwise_loss(
            pos_sym=neg_sym,
            neg_sym=pos_sym,
            margin=margin
        )

    def sym_gen():
        return transE_body(
                mx.sym.Variable('sub'),
                mx.sym.Variable('obj'),
                mx.sym.Variable('rel'),
            )


    sym = sym_gen()
    pair_sym = pairwise_sym_gen(margin)

    from longling.framework.ML.MXnet.mx_model import PairWiseModule
    mod = PairWiseModule(
        symbol=sym,
        data_names=['sub', 'rel', 'obj'],
        label_names=[],
        pairwise_symbol=pair_sym,
        pair_data_names=['pos_sub', 'pos_rel', 'pos_obj', 'neg_sub', 'neg_rel', 'neg_obj'],
        logger=module_logger,
        context=ctx,
    )

    # plot_network(
    #     nn_symbol=sym,
    #     save_path=model_dir + "plot/network",
    #     shape=form_shape(train_data),
    #     node_attrs={"fixedsize": "false"},
    #     view=True
    # )
    ############################################################################

    ############################################################################
    # fitting
    speedometer = Speedometer(
        batch_size=batch_size,
        frequent=100,
        # metrics=cross_entropy,
        logger=train_logger,
    )
    with tqdm_speedometer() as tins, TimeMonitor() as time_monitor:
        mod.pairwise_fit(
            train_data=train_data,
            eval_metric=[PairwiseMetric()],
            # eval_data=test_data,
            # validation_metric=[PRF(), Accuracy(), cross_entropy],
            # eval_end_callback=ClassificationLogValidationMetricsCallback(
            #     time_monitor,
            #     logger=validation_logger,
            #     logfile=validation_result_file,
            # ),
            begin_epoch=begin_epoch,
            num_epoch=num_epoch,
            optimizer='rmsprop',
            allow_missing=True,
            optimizer_params={'learning_rate': 0.0005},
            batch_end_callback=[tins],
            epoch_end_callback=[
                # mx.callback.do_checkpoint(model_dir + model_name),
                TqdmEpochReset(tins, "training"),
            ],
            monitor=time_monitor,
        )
    ############################################################################


if __name__ == '__main__':
    transE()

