# coding:utf-8
import copy
import json
import logging
import time
from collections import namedtuple

import mxnet as mx
import numpy as np

from longling.framework import get_text_cnn_symbol
from longling.framework import get_w2v


def to_unicode(s):
    return s.decode('utf-8') if isinstance(s, str) else s

def pad_sentences(sentences, padding_word="</s>", sentence_size=25):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    # sequence_length = max(len(x) for x in sentences)
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

CNNModel = namedtuple('CNNModel', ['cnn_exec', 'symbol', 'data', 'label', 'param_blocks'])

class TextCNNScorer(object):

    def __init__(self,prop):
        self.prop = copy.deepcopy(prop)
        self.init_scorer()

    def init_scorer(self):
        self.batch_size = self.prop.get('batch_size', 300)
        #ctx = mx.gpu(9)
        ctx = mx.cpu(0)

        sentence_size = self.prop['sentence_size']
        vec_size = self.prop['vec_size']
        location_vec = self.prop['location_vec']
        location_model = self.prop['location_model']
        sm = get_text_cnn_symbol(
            sentence_size=sentence_size,
            batch_size=self.batch_size,
            vec_size=vec_size,
            dropout=0.0,
            num_label=self.prop.get('label', 2),
            filter_list=self.prop.get('filter_list', [1, 2, 3, 4]),
            num_filter=self.prop.get('num_filter', 60)
        )

        input_shapes = {'data': (self.batch_size, 1, sentence_size, vec_size)}
        arg_shape, out_shape, aux_shape = sm.infer_shape(**input_shapes)
        arg_names = sm.list_arguments()

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
            else:
                arg_arrays.append(arg_dict[name])


        cnn_exec = sm.bind(ctx=ctx,args=arg_arrays)

        data = cnn_exec.arg_dict['data']
        label = cnn_exec.arg_dict['softmax_label']

        self.scorer = CNNModel(cnn_exec=cnn_exec, symbol=sm, data=data, label=label, param_blocks=None)

        self.vec = get_w2v(location_vec)

    def conv(self, sentences): # 分词之后的结果
        w2v = self.vec
        sentences = [sentence.lower().split() for sentence in sentences]
        sentences_padded = pad_sentences(sentences)
        x_vec = []
        for sent in sentences_padded:
            vec = []
            for word in sent:
                if word in w2v:
                    vec.append(w2v[word])
                else:
                    vec.append(w2v['</s>'])
                #print word,vec[-1][0]
            x_vec.append(vec)
        x_vec = np.array(x_vec)

        return x_vec

    def get_score(self, sentences):
        x_vec = self.conv(sentences)
        x_vec = np.reshape(x_vec, (x_vec.shape[0], 1, x_vec.shape[1], x_vec.shape[2]))
        self.scorer.data[:] = x_vec
        self.scorer.cnn_exec.forward(is_train=False)
        res = self.scorer.cnn_exec.outputs[0].asnumpy()
        return [r.argmax() for r in res]

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
        s = self.get_score(sentences)
        rst = s[:n_sent]
        t2 = time.time()
        logging.info("process titles: %s, cost: %s"%(n_sent, t2-t1))
        return rst

    def process(self, data):
        sentences = []
        rows = data.get('rows')
        if rows:
            rows = json.loads(rows)
            for row in rows:
                cut_title = to_unicode(row.get('cut_text', ''))
                sentences.append(cut_title)
        scores = self._process(sentences)
        if not scores:
            return {'msg': 'fail. none sentences'}
        return {'score': json.dumps(scores), 'msg': 'success'}

