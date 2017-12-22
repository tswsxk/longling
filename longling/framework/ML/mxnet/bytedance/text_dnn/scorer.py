#coding:utf-8

import sys
import json
sys.path.insert(0,"/opt/tiger/text_lib/")

import gpu_mxnet as mx

#import mxnet as mx
import numpy as np

from text_dnn import load_text_dnn_symbol,DNNModel

from data_store import get_vocab

def pad_sentences(sentences, padding_word="</s>", sentence_size=25):
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


class TextDNNScorer(object):
    
    def __init__(self, params):
        batch_size = 1
        

        location_symbol = params['location_symbol']
        location_model = params['location_model']
        location_vec = params['location_vec']
        location_size = params['location_size']
        id2v, self.w2id = get_vocab(location_vec)

        ctx = mx.cpu()
        sm = load_text_dnn_symbol(location_symbol, batch_size, is_train=False)


        with open(location_size) as fin:
            model_size = json.loads(fin.readline())
        sentence_size = model_size['sentence_size']
        

        input_shapes = {'data': (batch_size, sentence_size)}
        arg_shape, out_shape, aux_shape = sm.infer_shape(**input_shapes)
        arg_names = sm.list_arguments()

        p = mx.nd.load(location_model)
        arg_dict = {}
        for k,v in p.items():
            print k
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


        dnn_exec = sm.bind(ctx=ctx, args=arg_arrays)

        data = dnn_exec.arg_dict['data']
        label = dnn_exec.arg_dict['softmax_label']

        self.scorer = DNNModel(dnn_exec=dnn_exec, symbol=sm, data=data, label=label, param_blocks=None)
        self.sentence_size = sentence_size

    @staticmethod
    def check_paramters(params):
        for n in ['location_symbol', 'location_model', 'location_vec', 'sentence_size']:
            if n not in params:
                return False
        return True

    def conv(self, sentence): # 分词之后的结果
        sentences = [sentence]
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

    def get_score(self,sentence):
        x_vec = self.conv(sentence)

        self.scorer.data[:] = x_vec
        self.scorer.dnn_exec.forward(is_train=False)
        res = self.scorer.dnn_exec.outputs[0].asnumpy()
        return float(res[0][1])


if __name__ == '__main__':

    print 'x'

    location_model = ''
    location_model = '../collect_message/model/dnn-0001.params.cpu'
    location_symbol = "../collect_message/model/dnn-symbol.json"
    location_vec = '../ugc.vec'
    location_size = "../collect_message/model/size.txt"
    
    sentence = u'多少 人 是 被 封面 骗 进来 的'
    # vec_size = 80
    sentence_size = 25
    scorer = TextDNNScorer({
        'location_model':location_model, 
        'location_symbol':location_symbol, 
        'location_vec':location_vec,
        'location_size':location_size
    })

    res = scorer.get_score(sentence)
    print res

