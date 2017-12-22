# coding:utf-8
import copy
import logging
import time
from collections import namedtuple

import mxnet as mx
import numpy as np

from longling.framework import get_text_cnn_symbol_without_loss
from longling.framework import get_w2v

CNNModel = namedtuple('CNNModel', ['cnn_exec', 'symbol', 'data', 'label', 'param_blocks'])

##################################################################################################
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
##################################################################################################
def get_text_cnn_symbol(sentence_size, vec_size, batch_size, vocab_size=None,
                        num_label=2, filter_list=[1, 2, 3, 4], num_filter=60, dropout=0.0):

    fc = get_text_cnn_symbol_without_loss(sentence_size, vec_size, batch_size, vocab_size, num_label, filter_list, num_filter, dropout)
    sm = mx.symbol.MakeLoss(mx.symbol.max(fc, axis=1))
    return sm
##################################################################################################
class TextCNNHighlighter(object):

    def __init__(self,prop):
        self.prop = copy.deepcopy(prop)
        self.init_scorer()

    def init_scorer(self):
        self.batch_size = self.prop.get('batch_size', 300)
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

        args_grad = {}
        args_grad['data'] = mx.nd.zeros(input_shapes['data'], ctx)

        self.args_grad = args_grad

        arg_dict = {}
        for k,v in p.items():
            if k.startswith('arg:'):
                k = k[4:]
                arg_dict[k] = v

        arg_arrays = []
        for name, s in zip(arg_names, arg_shape):
            if name in ['data']:
                arg_arrays.append(mx.nd.zeros(s, ctx))
            else:
                arg_arrays.append(arg_dict[name])


        cnn_exec = sm.bind(ctx=ctx,args=arg_arrays, args_grad=args_grad)

        data = cnn_exec.arg_dict['data']

        self.higlighter = CNNModel(cnn_exec=cnn_exec, symbol=sm, data=data, label=None, param_blocks=None)

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

    def get_grad(self, sentences, map_range, center_tag=False, two_dir_tag=False, tail=0, empty_shift=0, valid_pad=0):
        x_vec = self.conv(sentences)
        x_vec = np.reshape(x_vec, (x_vec.shape[0], 1, x_vec.shape[1], x_vec.shape[2]))
        self.higlighter.data[:] = x_vec

        self.higlighter.cnn_exec.forward(is_train=True)
        self.higlighter.cnn_exec.backward()
        grad_map = self.args_grad['data'].asnumpy() * x_vec

        keyword_maps = []

        for idx, data in enumerate(grad_map):
            data = data[0]
            if (data == 0).all():
                continue
            else:
                keyword_map = []
                sentence = sentences[idx].lower().split()
                for row in data:
                    keyword_map.append(np.sum(row))

                leave_idx = set(list(np.nonzero(keyword_map)[0]))
                leave_idx.update(set(range(len(sentence))))
                if len(leave_idx) <= len(sentence):
                    max_extra_valid_idx = len(sentence) - 1
                else:
                    max_extra_valid_idx = max(filter(lambda x: x >= len(sentence), leave_idx))

                if empty_shift:
                    shift_array = keyword_map[len(sentence): max_extra_valid_idx + 1]
                    if not shift_array:
                        shift_value = 0
                    elif empty_shift == 1:
                        # average
                        shift_value = np.average(shift_array)
                    elif empty_shift == 2:
                        # max
                        shift_value = np.max(shift_array)
                    elif empty_shift == 3:
                        # min
                        shift_value = np.min(shift_array)
                    else:
                        shift_value = 0

                    keyword_map = np.add(keyword_map, -shift_value)

                if tail:
                    if tail == 1:
                        # only leave the word in sentence
                        keyword_map = keyword_map[:len(sentence)]
                        pass
                    elif tail == 2:
                        # extra give a 0
                        keyword_map = list(keyword_map[:len(sentence)])
                        keyword_map.append(0)
                        keyword_map = np.array(keyword_map)
                    else:
                        # leave the valid num
                        keyword_map = np.array([keyword_map[lidx] for lidx in leave_idx])

                if map_range is not None:
                    def range_map(keyword_map):
                        max_num = np.max(keyword_map)
                        min_num = np.min(keyword_map)
                        if center_tag and max_num * min_num < 0:
                            norm = np.max(np.abs([max_num, min_num]))
                            mul_norm = float(max(map_range)) / norm
                            keyword_map = np.multiply(keyword_map, mul_norm)
                            return keyword_map

                        elif two_dir_tag and max_num * min_num < 0:
                            pos_norm, neg_norm = np.abs([max_num, min_num])
                            pos_mul_norm, neg_mul_norm = float(max(map_range)) / pos_norm, float(max(map_range)) / neg_norm
                            keyword_map = map(lambda x: x * pos_mul_norm if x >= 0 else x * neg_mul_norm, keyword_map)
                            keyword_map = np.array(keyword_map)
                            return keyword_map
                        else:
                            norm = max_num - min_num
                            keyword_map = np.add(keyword_map, -min_num)
                            keyword_map *= (max(map_range) - min(map_range))
                            if norm != 0:
                                keyword_map /= norm
                            else:
                                keyword_map = np.array(len(keyword_map) * [max(map_range)])
                            return np.add(keyword_map, min(map_range))
                    keyword_map = range_map(keyword_map)
                if valid_pad:
                    # leave the valid num
                    if valid_pad == 1:
                        threshold = 1 if max(map_range) == 256 else 1e-20
                        valid_index = [vi + len(sentence) for vi, value in enumerate(keyword_map[len(sentence): max_extra_valid_idx + 1]) if abs(value) >= threshold]
                        if valid_index:
                            leave_idx = filter(lambda x: x <= max(valid_index), leave_idx)
                            keyword_map = [keyword_map[lidx] for lidx in leave_idx]
                            sentence.extend([""] * (len(keyword_map) - len(sentence)))
                        else:
                            keyword_map = keyword_map[:len(sentence)]

                    else:
                        keyword_map = [keyword_map[lidx] for lidx in leave_idx]
                        sentence.extend([""] * (len(keyword_map) - len(sentence)))
                else:
                    keyword_map = keyword_map[:len(sentence)]
                keyword_map = [(sentence[i], keyword_map[i]) for i in range(len(keyword_map))]
                keyword_maps.append(keyword_map)

        return keyword_maps

    def process(self, sentences, map_range=None, center_tag=False, two_dir_tag=False, tail=0, empty_shift=0, valid_pad=0):
        t1 = time.time()
        n_sent = len(sentences)
        rst = []
        if not sentences:
            return rst
        elif len(sentences) <= self.batch_size:
            sentences += [''] * (self.batch_size-len(sentences))
        else:
            return rst
        s = self.get_grad(sentences, map_range, center_tag, two_dir_tag, tail, empty_shift, valid_pad)
        rst = s[:n_sent]

        t2 = time.time()
        logging.info("process titles: %s, cost: %s" % (n_sent, t2 - t1))

        return rst
