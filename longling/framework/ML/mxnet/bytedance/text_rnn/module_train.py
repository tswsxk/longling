# -*- coding: utf-8 -*-
import sys

# sys.path.insert(0, "/data01/home/tongshiwei/repos/text_env/env/local/lib/python2.7/site-packages/mxnet-0.9.5-py2.7.egg/")
sys.path.insert(0, "/opt/tiger/nlp/text_env/env/lib/python2.7/site-packages/mxnet-0.9.5-py2.7.egg")
import mxnet as mx
import numpy as np
import random
import bisect
import codecs
import json
import argparse

import logging

reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.NOTSET, datefmt='%I:%M:%S')

from lstm import lstm_unroll, bi_lstm_unroll
from bucket_io import BucketSentenceIter

from utils import to_unicode


def load_vocab_0(path):  # word:cnt
    rst = {}
    with codecs.open(path) as f:
        for line in f:
            l = json.loads(to_unicode(line))
            if len(l) == 2:
                rst[l[0]] = int(l[1])
    return rst


def load_vocab(path):  # word:vec
    rst = []  # keep the order
    with open(path) as f:
        for line in f:
            l = line.strip().split()
            if len(l) == 2:
                continue
            rst.append([l[0], map(float, l[1:])])
    return rst


def CalcSize(label, pred):
    label = label.T.reshape((-1,))
    return label.size


def CalcCorrectNum(label, pred):
    label = label.T.reshape((-1,))
    correct = 0
    for i in xrange(pred.shape[0]):
        a = label[i]
        b = pred[i][1]
        if b > 0.5 and a == 1:
            correct += 1
        if b < 0.5 and a == 0:
            correct += 1
    return correct

def CalcPrec_0(label, pred):
    tp = 0
    fp = 0
    for i in xrange(pred.shape[0]):
        a = label[i]
        b = pred[i][1]
        if b < 0.5 and a == 0:
            tp += 1
        if b < 0.5 and a == 1:
            fp += 1
    p = 1.0 * tp / (tp + fp) if tp + fp else 0
    return p

def CalcPrec_1(label, pred):
    tp = 0
    fp = 0
    for i in xrange(pred.shape[0]):
        a = label[i]
        b = pred[i][1]
        if b >= 0.5 and a == 1:
            tp += 1
        if b >= 0.5 and a == 0:
            fp += 1
    p = 1.0 * tp / (tp + fp) if tp + fp else 0
    return p

def CalcRecall_0(label, pred):
    tp = 0
    fn = 0
    for i in xrange(pred.shape[0]):
        a = label[i]
        b = pred[i][1]
        if b < 0.5 and a == 0:
            tp += 1
        if b >= 0.5 and a == 0:
            fn += 1
    r = 1.0 * tp / (tp + fn) if tp + fn else 0
    return r

def CalcRecall_1(label, pred):
    tp = 0
    fn = 0
    for i in xrange(pred.shape[0]):
        a = label[i]
        b = pred[i][1]
        if b >= 0.5 and a == 1:
            tp += 1
        if b < 0.5 and a == 1:
            fn += 1
    r = 1.0 * tp / (tp + fn) if tp + fn else 0
    return r


def fit(args):
    ############### train model ##########
    import conf
    print args
    batch_size = args.batch_size
    buckets = args.buckets
    num_hidden = args.num_hidden
    # num_embed = args.num_embed
    num_label = args.num_label
    epoch_nums = args.epoch_nums
    learning_rate = args.lr
    momentum = args.momentum
    network = args.network
    num_lstm_layer = 2 if network is "bilstm" else args.num_lstm_layer
    train_path = args.train_path
    test_path = args.test_path
    location_vec = args.location_vec
    model_prefix = args.model_prefix

    vocab = load_vocab(location_vec)
    print len(vocab)
    init_c = [('l%d_init_c' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_stats = init_c + init_h

    num_embed = len(vocab[0][1])
    print "num_embed", num_embed

    # save size text
    params = {}
    params['batch_size'] = args.batch_size
    params['num_label'] = args.num_label
    params['num_lstm_layer'] = args.num_lstm_layer
    params['location_vec'] = args.location_vec
    params['model_prefix'] = args.model_prefix
    params['epoch_num'] = args.epoch_nums
    params['num_hidden'] = args.num_hidden
    params['gpu'] = args.gpus
    params['num_embed'] = num_embed
    params['network'] = network
    params['buckets'] = buckets

    data_train = BucketSentenceIter(train_path, vocab, buckets, batch_size,
                                    init_stats, seperate_char='\n')

    data_val = BucketSentenceIter(test_path, vocab, buckets, batch_size,
                                  init_stats, seperate_char='\n')

    params['sentense_size'] = data_train.default_bucket_key / num_embed
    line = json.dumps(params)
    with open(args.location_size, mode='w') as f:
        f.write(line)

    # contexts = [mx.context.gpu(int(i)) for i in args.gpus.split(',')]
    gpu = args.gpus
    if isinstance(gpu, int):
        if gpu >= 0:
            contexts = mx.gpu(gpu)
        else:
            contexts = mx.cpu(gpu)
    elif isinstance(gpu, list):
        contexts = [mx.gpu(g) for g in gpu]
    else:
        contexts = [mx.gpu(int(g)) for g in gpu.split(',')]
    print num_embed, contexts

    def sym_gen(seq_len):
        seq_len = seq_len / num_embed  # fix
        rst = None
        if network == 'lstm':
            rst = lstm_unroll(num_lstm_layer,
                              seq_len,
                              len(vocab) + 1,
                              num_hidden=num_hidden,
                              num_embed=num_embed,
                              num_label=num_label,
                              dropout=0.2
                              )
        elif network == 'bilstm':
            rst = bi_lstm_unroll(seq_len,
                                 len(vocab) + 1,
                                 num_hidden=num_hidden,
                                 num_embed=num_embed,
                                 num_label=num_label,
                                 dropout=0.2
                                 )
        else:
            print "no match structure"
            exit(-1)
        # shape = dict([('data', (batch_size, seq_len * num_embed))] + init_stats + [('softmax_label', (batch_size, 1))])
        # shape = None
        # mx.viz.plot_network(rst, shape=shape).render("train_" + str(seq_len))
        # 可视化网络，检查网络结构用
        return rst, tuple(['data'] + [n[0] for n in init_stats]), ('softmax_label',)


    model = mx.mod.BucketingModule(context=contexts, sym_gen=sym_gen, default_bucket_key=data_train.default_bucket_key)

    eval_metrics = []
    eval_metrics.append('accuracy')
    eval_metrics.append('f1')
    eval_metrics.append(mx.metric.np(CalcSize, name='size'))
    eval_metrics.append(mx.metric.np(CalcCorrectNum, name='correct_num'))
    eval_metrics.append(mx.metric.np(CalcPrec_0, name='P0'))
    eval_metrics.append(mx.metric.np(CalcRecall_0, name='R0'))
    eval_metrics.append(mx.metric.np(CalcPrec_1, name='P1'))
    eval_metrics.append(mx.metric.np(CalcRecall_1, name='R1'))
    eval_metrics = mx.metric.create(eval_metrics)
    from epoch_end_callback import Msger, epochCallbacker, evalEndCallbacker

    msger = Msger(model_prefix)

    model.fit(
        train_data=data_train,
        eval_data=data_val,
        eval_metric=eval_metrics,
        epoch_num=epoch_nums,
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        optimizer_params=(('learning_rate', learning_rate), ('momentum', momentum), ('wd', 0.0001),),
        epoch_end_callback=epochCallbacker(msger, eval_metrics, data_val),
        eval_end_callback=evalEndCallbacker(msger)
    )


def parse_args():
    parser = argparse.ArgumentParser(description='train lstm')
    parser.add_argument('--gpus', type=str, default='0',
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--num-hidden', type=int, default=128,
                        help='the num hidden')
    parser.add_argument('--num-embed', type=int, default=64,
                        help='the num embed')
    parser.add_argument('--num-lstm-layer', type=int, default=1,
                        help='the num lstm layer')
    parser.add_argument('--num-epochs', type=int, default=2,
                        help='the number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='the momentum of sgd')
    parser.add_argument('--num-label', type=int, default=2,
                        help='the number of labels')
    parser.add_argument('--network', type=str, default='lstm',
                        help='the number of labels')

    # data
    parser.add_argument('--train-path', type=str, required=True,
                        help='the train path')
    parser.add_argument('--test-path', type=str, required=True,
                        help='the test path')
    parser.add_argument('--vocab-path', type=str, required=True,
                        help='the vocab path')
    # save model
    parser.add_argument('--model-prefix', type=str, required=True,
                        help='the prefix of the model to save')
    # parser.add_argument('--load-epoch', type=int,
    #                    help="load the model on an epoch using the model-prefix")
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    args = parse_args()
    fit(args)
