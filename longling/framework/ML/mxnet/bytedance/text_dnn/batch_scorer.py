#coding:utf-8

import sys
sys.path.insert(0,"/opt/tiger/text_lib/")


import gpu_mxnet as mx

#import mxnet as mx
import numpy as np
import json
import time

import logging

from text_dnn import load_text_dnn_symbol, DNNModel

from data_store import get_vocab
import gpu_mgr

def to_unicode(s):
    return s.decode('utf-8') if isinstance(s, str) else s


def pad_sentences(sentences, padding_word="</s>",sentence_size=25):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    #sequence_length = max(len(x) for x in sentences)
    sequence_length = sentence_size
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) < sequence_length:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:sequence_length]
        padded_sentences.append(new_sentence)
    return padded_sentences

class BatchTextDNNScorer(object):
    def __init__(self, sentence_size, location_vec, location_model, location_symbol, batch_size=128, is_multilabel=False, gpu=None):
        self.batch_size = batch_size

        if gpu == None:
            try:
                gpu = gpu_mgr.get_available_gpu()
                ctx = mx.gpu(gpu)
            except:
                ctx = mx.cpu()

        sm = load_text_dnn_symbol(location_symbol, self.batch_size)

        input_shapes = {'data': (self.batch_size, sentence_size)}
        arg_shape, out_shape, aux_shape = sm.infer_shape(**input_shapes)
        arg_names = sm.list_arguments()

        id2v, self.w2id = get_vocab(location_vec)
        
        # 读入模型参数
        p = mx.nd.load(location_model)
        arg_dict = {}
        for k,v in p.items():
            if k.startswith('arg:'):
                k = k[4:]
                arg_dict[k] = v
        arg_arrays = []
        for name, s in zip(arg_names, arg_shape):
            if name in ['softmax_label', 'data']:
                arg_arrays.append(mx.nd.zeros(s, ctx))
            elif name == "embedding_weight":
                arg_arrays.append(mx.nd.array(id2v))
            else:
                arg_arrays.append(arg_dict[name])


        dnn_exec = sm.bind(ctx=ctx,args=arg_arrays)

        data = dnn_exec.arg_dict['data']
        label = dnn_exec.arg_dict['softmax_label']

        self.scorer = DNNModel(dnn_exec=dnn_exec, symbol=sm, data=data, label=label, param_blocks=None)

        self.sentence_size = sentence_size
        self.multilabel = is_multilabel

    def conv(self, sentences): # 分词之后的结果
      
        sentences = [sentence.lower().split() for sentence in sentences]
        sentences_padded = pad_sentences(sentences,sentence_size=self.sentence_size)
        x_vec = []
        for sent in sentences_padded:
            vec = []
            for word in sent:
                if word in self.w2id:
                    vec.append(self.w2id[word])
                else:
                    vec.append(self.w2id['</s>'])
           
            x_vec.append(vec)
        x_vec = np.array(x_vec)

        return x_vec

    def get_score(self, sentences):
        x_vec = self.conv(sentences)
        # x_vec = np.reshape(x_vec, (x_vec.shape[0], 1, x_vec.shape[1], x_vec.shape[2]))
        self.scorer.data[:] = x_vec
        self.scorer.dnn_exec.forward(is_train=False)
        res = self.scorer.dnn_exec.outputs[0].asnumpy()
        return [float(r[1]) for r in res]

    def multi_get_score(self, sentences):
        x_vec = self.conv(sentences)
        # x_vec = np.reshape(x_vec, (x_vec.shape[0], 1, x_vec.shape[1], x_vec.shape[2]))
        self.scorer.data[:] = x_vec
        self.scorer.dnn_exec.forward(is_train=False)
        res = self.scorer.dnn_exec.outputs[0].asnumpy()
        ms = []
        for r in res:
            max_score = r[0]
            max_mark = 0
            for i in range(len(r)):
                if r[i] > max_score:
                    max_mark = i
                    max_score = r[i]
            ms.append((str(max_mark),max_score))
        return ms

    def _process(self, sentences):
        t1 = time.time()
        n_sent = len(sentences)
        rst = []
        if not sentences:
            return rst
        elif len(sentences) <= self.batch_size:
            sentences += [''] * (self.batch_size-len(sentences))
        else:
            return rst
        
        if not self.multilabel:
            s = self.get_score(sentences)
        elif self.multilabel:
            s = self.multi_get_score(sentences)

        rst = s[:n_sent]
        t2 = time.time()
        #logging.info("process sentences: %s, cost: %s"%(n_sent, t2-t1))
        return rst

    def get_scores(self,sentences):
        return self._process(sentences)

    def process(self, data):
        sentences = []
        rows = data.get('rows')
        if rows:
            rows = json.loads(rows)
            for row in rows:
                cut_title = to_unicode(row.get('sentence', ''))
                sentences.append(cut_title)
        scores = self._process(sentences)
        if not scores:
            return {'msg': 'fail. none sentences'}
        return {'scores': json.dumps(scores), 'msg': 'success'}



# class BatchTextDNNScorer(object):

#     def __init__(self,sentence_size, vec_size,
#                     location_vec,location_model,location_symbol,batch_size=128):
#         self.batch_size = batch_size
#         ctx = mx.gpu(10)
#         #ctx = mx.cpu(0)

#         #sm = get_text_cnn_symbol(sentence_size,vec_size,self.batch_size,dropout=0.0)
#         sm = load_text_dnn_symbol(location_symbol,self.batch_size)

#         input_shapes = {'data': (self.batch_size, vec_size)}
#         arg_shape, out_shape, aux_shape = sm.infer_shape(**input_shapes)
#         arg_names = sm.list_arguments()

#         p = mx.nd.load(location_model)
#         arg_dict = {}
#         for k,v in p.items():
#             if k.startswith('arg:'):
#                 k = k[4:]
#                 arg_dict[k] = v
#         arg_arrays = []
#         for name, s in zip(arg_names, arg_shape):
#             if name in ['softmax_label', 'data']:
#                 arg_arrays.append(mx.nd.zeros(s, ctx))
#             else:
#                 arg_arrays.append(arg_dict[name])


#         dnn_exec = sm.bind(ctx=ctx,args=arg_arrays)

#         data = dnn_exec.arg_dict['data']
#         label = dnn_exec.arg_dict['softmax_label']

#         self.scorer = DNNModel(dnn_exec=dnn_exec, symbol=sm, data=data, label=label, param_blocks=None)

#         self.vec = get_w2v(location_vec)
#         self.sentence_size = sentence_size

#     def conv(self,sentences): # 分词之后的结果
#         w2v = self.vec
#         zero = np.zeros_like(w2v['</s>'])
#         sentences = [sentence.lower().split() for sentence in sentences]
#         sentences_padded = pad_sentences(sentences,sentence_size=self.sentence_size)
#         x_vec = []
#         for sent in sentences_padded:
#             vec = []
#             for word in sent:
#                 if word in w2v:
#                     vec.append(w2v[word])
#                 else:
#                     # vec.append(w2v['</s>'])
#                     vec.append(zero)
#                 #print word,vec[-1][0]

#             # if len(vec) == 0:
#             #     vec = [w2v['</s>']]
#             # vec = sum(map(np.array, vec))
#             x_vec.append(vec)
#         x_vec = np.array(x_vec)
#         x_vec = np.sum(x_vec, axis=1)

#         return x_vec

#     def get_score(self, sentences):
#         x_vec = self.conv(sentences)
#         # x_vec = np.reshape(x_vec, (x_vec.shape[0], 1, x_vec.shape[1], x_vec.shape[2]))
#         self.scorer.data[:] = x_vec
#         self.scorer.dnn_exec.forward(is_train=False)
#         res = self.scorer.dnn_exec.outputs[0].asnumpy()
#         return [float(r[1]) for r in res]

#     def _process(self, sentences):
#         t1 = time.time()
#         n_sent = len(sentences)
#         rst = []
#         if not sentences:
#             return rst
#         elif len(sentences) <= self.batch_size:
#             sentences += [''] * (self.batch_size-len(sentences))
#         else:
#             return rst
#         s = self.get_score(sentences)
#         rst = s[:n_sent]
#         t2 = time.time()
#         #logging.info("process sentences: %s, cost: %s"%(n_sent, t2-t1))
#         return rst

#     def get_scores(self,sentences):
#         return self._process(sentences)

#     def process(self, data):
#         sentences = []
#         rows = data.get('rows')
#         if rows:
#             rows = json.loads(rows)
#             for row in rows:
#                 cut_title = to_unicode(row.get('sentence', ''))
#                 sentences.append(cut_title)
#         scores = self._process(sentences)
#         if not scores:
#             return {'msg': 'fail. none sentences'}
#         return {'scores': json.dumps(scores), 'msg': 'success'}

