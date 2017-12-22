# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "/opt/tiger/nlp/text_env/env/lib/python2.7/site-packages/mxnet-0.9.5-py2.7.egg")
import mxnet as mx
import numpy as np
import codecs
import json

import logging
reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

from rnn_model import LSTMInferenceModel, BiLSTMInferenceModel

from utils import to_unicode


# make input from char
def MakeInput(char, vocab, num_embed, arr):
    idx = vocab[char]
    tmp = np.zeros((1,num_embed))#fix
    tmp[0] = idx
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
        self.idx_gpu = params['idx_gpu']

        self.model = self.load_model()
        #self.id2vocab = {i: v for v, i in self.vocab.iteritems()}

    def load_model(self):
        model_prefix = self.model_prefix
        epoch_num = self.epoch_num
        idx_gpu = self.idx_gpu

        _, arg_params, __ = mx.model.load_checkpoint(model_prefix, epoch_num)
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
                                   dropout=0
                                )
        return model

    def process(self, s):
        input_ndarray = mx.nd.zeros((1,self.num_embed))#fix
        prob = 0
        for i in range(len(s)):
            new_seq = True if i == 0 else False
            term = s[i]
            if term not in self.vocab:
                continue
            # print term
            MakeInput(term, self.vocab, self.num_embed, input_ndarray)
            fw = self.model.forward(input_ndarray, new_seq)
            prob = fw['prob'][0][1]
        return prob


class BiLstmPredict():
    def __init__(self, params):
        self.num_hidden = params['num_hidden']
        self.num_embed = params['num_embed']
        self.num_label = params['num_label']
        self.num_lstm_layer = params['num_lstm_layer']
        self.location_vec = params['location_vec']
        self.vocab = load_vocab(self.location_vec)

        self.model_prefix = params['model_prefix']
        self.epoch_num = params['epoch_num']
        self.idx_gpu = params['idx_gpu']
        self.seq_len = params.get('seq_len', 0)

        #self.id2vocab = {i: v for v, i in self.vocab.iteritems()}
        self.seq_model = {}

    def load_model(self, seq_len):
        model_prefix = self.model_prefix
        epoch_num = self.epoch_num
        idx_gpu = self.idx_gpu

        _, arg_params, __ = mx.model.load_checkpoint(model_prefix, epoch_num)
        if idx_gpu >= 0:
            ctx = mx.gpu(idx_gpu)
        else:
            ctx = mx.cpu()

        model = BiLSTMInferenceModel(seq_len,
                                     len(self.vocab) + 1,
                                     num_hidden=self.num_hidden,
                                     num_embed=self.num_embed,
                                     num_label=self.num_label,
                                     arg_params=arg_params,
                                     ctx=ctx,
                                     dropout=0.)
        return model


    def process(self, s):
        seq_len = len(s)
        if seq_len not in self.seq_model:
            self.seq_model[seq_len] = self.load_model(seq_len)
        model = self.seq_model[seq_len]
        data = []#fix
        ss = [k for k in s if k in self.vocab]
        for i in range(len(ss)):
            data.extend(self.vocab.get(ss[i]))
        data.extend([0]*self.num_embed*(len(s)-len(ss)))
        data = [data]
        data = mx.nd.array(data)
        fw = model.forward(data, True)
        prob = fw['prob'][0][1]
        return prob
