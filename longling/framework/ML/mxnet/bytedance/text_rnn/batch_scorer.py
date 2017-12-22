# -*- coding: utf-8 -*-
import sys
import os
#sys.path.insert(0, "/opt/tiger/nlp/text_env/env/lib/python2.7/site-packages/mxnet-0.9.5-py2.7.egg")

import sys
sys.path.insert(0,"/opt/tiger/text_lib/")


import os
if os.getenv('MXNET_TYPE') == 'gpu_mxnet':
    import gpu_mxnet as mx
else:
    import mxnet as mx

import numpy as np
import codecs
import json

import logging
reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

from rnn_model import LSTMInferenceModel, BiLSTMInferenceModel

from utils import to_unicode

import json
import conf

# make input from char
def MakeInput(chars, vocab, num_embed, arr):
    idx = [vocab[char] for char in chars]
    tmp = np.zeros((len(chars),num_embed))#fix
    tmp[:] = idx
    arr[:] = tmp

def load_vocab_0(path):
    rst = {}
    with codecs.open(path) as f:
        for line in f:
            l = json.loads(to_unicode(line))
            if len(l) == 2:
                rst[l[0]] = int(l[1])
    return rst

def load_vocab(path):
    rst = {}
    with codecs.open(path) as f:
        for line in f:
            l = line.strip().split()
            if len(l) == 2:
                continue
            rst[l[0].decode('utf-8')] = map(float,l[1:])
            rst[l[0]] = map(float,l[1:])
    return rst

# Evaluation
def Perplexity(label, pred):
    loss = 0.0
    for i in xrange(len(pred)):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss/len(label))


class LstmPredict():
    def __init__(self, params):
        self.num_hidden = params['num_hidden']
        self.num_embed = params['num_embed']
        self.num_label = params['num_label']
        self.num_lstm_layer = params['num_lstm_layer']
        self.location_vec = params['location_vec']
        self.vocab = load_vocab(self.location_vec)

        self.model_prefix = params['model_prefix']
        self.epoch_num = params['epoch_num']
        self.idx_gpu = params['gpu']
        self.batch_size = params['batch_size']

        self.model = self.load_model()
        #self.id2vocab = {i: v for v, i in self.vocab.iteritems()}

    def load_model(self):
        model_prefix = self.model_prefix
        epoch_num = self.epoch_num
        idx_gpu = self.idx_gpu

        _, arg_params, __ = mx.model.load_checkpoint(model_prefix, epoch_num)
        if isinstance(idx_gpu, list):
            idx_gpu = idx_gpu[0]
        if idx_gpu >= 0:
            ctx = mx.gpu(idx_gpu)
        else:
            ctx = mx.cpu()
        model = LSTMInferenceModel(self.num_lstm_layer,
                                   len(self.vocab) + 1,
                                   num_hidden=self.num_hidden,
                                   num_embed=self.num_embed,
                                   num_label=self.num_label,
                                   arg_params=arg_params,
                                   ctx=ctx,
                                   dropout=0.2,
                                   batch_size=self.batch_size
                                )
        return model

    def _process(self, sentences):
        if not sentences:
            return []
        elif len(sentences) < self.batch_size:
            sentences += [''] * (self.batch_size-len(sentences))
        new_sentences = []
        for sentence in sentences:
            new_sentences.append([term for term in sentence if term in self.vocab])

        sentences = new_sentences
        input_ndarray = mx.nd.zeros((len(sentences), self.num_embed))  # fix
        input_sentences_len = [len(sentence) for sentence in sentences]
        probs = [0 for _ in input_sentences_len]

        for i in range(max(input_sentences_len)):
            new_seq = True if i == 0 else False
            terms = []
            for sentence in sentences:
                if i >= len(sentence):
                    if len(sentence) == 0:
                        terms.append('</s>')
                    else:
                        terms.append(sentence[len(sentence) - 1])
                else:
                    terms.append(sentence[i])

            # print term
            MakeInput(terms, self.vocab, self.num_embed, input_ndarray)
            fw = self.model.forward(input_ndarray, new_seq)
            for sen_idx in range(len(sentences)):
                if i < input_sentences_len[sen_idx]:
                    probs[sen_idx] = fw['prob'][sen_idx][1]
        return probs

    def process(self, sentences):
        return self._process(sentences)


def get_x(m):
    return m['data']

def process_batch_score_old(location_dat, location_res, location_vec, location_model, epoch, max_cnt=-1, idx_gpu=0):
    # 存在问题，打分会和训练过程中评价不一致，请勿使用
    # 可能原因：
    # 1. 训练数据格式
    # 2. 网络结构
    model_dir = os.path.dirname(location_model)
    model_prefix = os.path.join(model_dir, 'rnn')
    location_size = os.path.join(model_dir, 'size.txt')
    # load size text
    with open(location_size) as fin:
        params = json.load(fin)
    params['epoch_num'] -= 1
    batch_size = params['batch_size']
    # batch_size = conf.PREDICT_BATCH_SIZE
    # params = {
    #     "num_hidden": conf.NUM_HIDDEN,
    #     "num_embed": conf.NUM_EMBED,
    #     "num_label": conf.NUM_LABEL,
    #     "num_lstm_layer": conf.NUM_LSTM_LAYER,
    #     "location_vec": location_vec,

    #     "model_prefix": model_prefix,
    #     'epoch_num': epoch - 1,
    #     'idx_gpu': idx_gpu,
    #     'batch_size': batch_size
    #   }
    scorer = LstmPredict(params)

    f_t = open(location_dat, mode='r')
    f_r = open(location_res, mode='w')
    print "begin give score"
    cnt = 0
    ms = []
    import time
    st = time.time()
    print "------------batch_score------------"
    for line in f_t:
        cnt += 1
        if max_cnt > 0 and cnt > max_cnt:
            break
        if cnt % 10000 ==0:
            et = time.time()
            print "score completed % s, cost %s" % (cnt, et - st)
            st = et
        m = json.loads(line)
        ms.append(m)
        if len(ms) == batch_size:
            sentences = []
            for i in range(batch_size):
                sentences.append(get_x(ms[i]))
            res = scorer.process(sentences)
            for i in range(batch_size):
                ms[i]['score'] = float(res[i])
            for m in ms:
                line = json.dumps(m, ensure_ascii=False)
                f_r.write(line.encode('utf-8') + '\n')
            ms = []

    sentences = []
    cc = len(ms)
    for i in range(cc):
        sentences.append(get_x(ms[i]))
    res = scorer.process(sentences)
    for i in range(cc):
        ms[i]['score'] = float(res[i])
    for m in ms:
        line = json.dumps(m, ensure_ascii=False)
        f_r.write(line.encode('utf-8') + '\n')

    f_t.close()
    f_r.close()

def process_batch_score(location_dat, location_res, location_vec, location_model, epoch, max_cnt=-1, idx_gpu=0, is_multilabel=False):
    def load_vocab(path):  # word:vec
        rst = []  # keep the order
        with open(path) as f:
            for line in f:
                l = line.strip().split()
                if len(l) == 2:
                    continue
                rst.append([l[0], map(float, l[1:])])
        return rst

    model_dir = os.path.dirname(location_model)
    model_prefix = os.path.join(model_dir, 'rnn')
    location_size = os.path.join(model_dir, 'size.txt')
    # load size text
    with open(location_size) as fin:
        params = json.load(fin)
    params['epoch_num'] -= 1
    batch_size = params['batch_size']
    # batch_size = conf.PREDICT_BATCH_SIZE
    # params = {
    #     "num_hidden": conf.NUM_HIDDEN,
    #     "num_embed": conf.NUM_EMBED,
    #     "num_label": conf.NUM_LABEL,
    #     "num_lstm_layer": conf.NUM_LSTM_LAYER,
    #     "location_vec": location_vec,

    #     "model_prefix": model_prefix,
    #     'epoch_num': epoch - 1,
    #     'idx_gpu': idx_gpu,
    #     'batch_size': batch_size
    #   }
    # scorer = LstmPredict(params)

    from bucket_io import BucketSentenceScoreIter
    from lstm import lstm_unroll, bi_lstm_unroll
    num_label = params['num_label']
    num_lstm_layer = params['num_lstm_layer']
    location_vec = params['location_vec']
    model_prefix = params['model_prefix']
    epoch_num = params['epoch_num']
    num_hidden = params['num_hidden']
    gpu = params['gpu']
    num_embed = params['num_embed']
    network = params['network']

    # if isinstance(gpu, int):
    #     if gpu >= 0:
    #         contexts = mx.gpu(gpu)
    #     else:
    #         contexts = mx.cpu(gpu)
    # elif isinstance(gpu, list):
    #     contexts = [mx.gpu(g) for g in gpu]
    # else:
    #     contexts = [mx.gpu(int(g)) for g in gpu.split(',')]

    contexts = mx.cpu()

    buckets = params['buckets']
    vocab = load_vocab(location_vec)
    init_c = [('l%d_init_c' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_stats = init_c + init_h
    print location_dat
    data_val = BucketSentenceScoreIter(location_dat, vocab, buckets, batch_size,
                                    init_stats, seperate_char='\n')
    def sym_gen(seq_len):
        seq_len = seq_len / params['num_embed']  # fix

        if network == 'bilstm':
            rst = bi_lstm_unroll(seq_len,
                                 len(vocab) + 1,
                                 num_hidden=num_hidden,
                                 num_embed=num_embed,
                                 num_label=num_label,
                                 dropout=0.2
                                 )
        else:
            # network == 'lstm'
            rst = lstm_unroll(num_lstm_layer,
                              seq_len,
                              len(vocab) + 1,
                              num_hidden=num_hidden,
                              num_embed=num_embed,
                              num_label=num_label,
                              dropout=0.2
                              )

        return rst, tuple(['data'] + [n[0] for n in init_stats]), ('softmax_label',)


    model = mx.mod.BucketingModule(context=contexts, sym_gen=sym_gen, default_bucket_key=data_val.default_bucket_key)
    model.bind(data_val.provide_data, data_val.provide_label)
    model.load_params('%s-%04d.params.cpu' % (model_prefix, epoch_num))
    print "------------batch_score------------"
    f_r = open(location_res, mode='w')
    cnt = 0
    import time
    st = time.time()
    for pred, i_batch, batch in model.iter_predict(data_val):
        pred = [p[1] for p in pred[0].asnumpy()]
        rec = batch.rec
        cnt += batch.length
        for i in range(batch.length):
            rec[i]['score'] = float(pred[i])
            line = json.dumps(rec[i], ensure_ascii=False)
            f_r.write(line.encode('utf-8') + '\n')
        if i_batch % 100 == 0:
            et = time.time()
            print "score completed % s, cost %s" % (cnt, et - st)
            st = et
    no_score_rec = data_val.zero_len_rec()
    for rec in no_score_rec:
        rec['score'] = 0
        line = json.dumps(rec, ensure_ascii=False)
        f_r.write(line.encode('utf-8') + '\n')
    et = time.time()
    cnt += len(no_score_rec)
    print "score completed % s, cost %s" % (cnt, et - st)
    f_r.close()
