# coding: utf-8
# create by tongshiwei on 2017/10/22

import logging

import numpy as np
import mxnet as mx

from longling.framework.ML.mxnet.nn.shared.nn import get_model, get_embeding_model


def get_text_cnn_symbol_without_loss(sentence_size, vec_size, batch_size, vocab_size=None,
                                     num_label=2, filter_list=[1, 2, 3, 4], num_filter=60, dropout=0.0):
    input_x = mx.sym.Variable('data')

    if vocab_size is not None:
        embed_x = mx.sym.Embedding(data=input_x, input_dim=vocab_size, output_dim=vec_size, name="embedding")
        conv_input = mx.sym.Reshape(data=embed_x, shape=(batch_size, 1, sentence_size, vec_size))
    else:
        conv_input = input_x

    pooled_outputs = []
    for i, filter_size in enumerate(filter_list):
        convi = mx.sym.Convolution(data=conv_input, kernel=(filter_size, vec_size), num_filter=num_filter,
                                   name=('convolution%s' % i))
        relui = mx.sym.Activation(data=convi, act_type='relu', name=('activation%s' % i))
        pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(sentence_size - filter_size + 1, 1),
                               stride=(1, 1), name=('pooling%s' % i))
        pooled_outputs.append(pooli)

    total_filters = num_filter * len(filter_list)
    concat = mx.sym.Concat(dim=1, *pooled_outputs)
    h_pool = mx.sym.Reshape(data=concat, shape=(batch_size, total_filters))
    if dropout > 0.0:
        h_drop = mx.sym.Dropout(data=h_pool, p=dropout)
    else:
        h_drop = h_pool
    cls_weight = mx.sym.Variable('cls_weight')
    cls_bias = mx.sym.Variable('cls_bias')
    fc = mx.sym.FullyConnected(data=h_drop, weight=cls_weight, bias=cls_bias, num_hidden=num_label)

    return fc


def get_text_cnn_symbol(sentence_size, vec_size, batch_size, vocab_size=None,
                        num_label=2, filter_list=[1, 2, 3, 4], num_filter=60, dropout=0.0):
    input_y = mx.sym.Variable('label')
    fc = get_text_cnn_symbol_without_loss(sentence_size, vec_size, batch_size, vocab_size, num_label, filter_list,
                                          num_filter, dropout)
    sm = mx.sym.SoftmaxOutput(data=fc, label=input_y, name='softmax')

    return sm


def get_text_cnn_model(ctx, cnn_symbol, embedding, sentence_size, batch_size, return_grad=False,
                       checkpoint=None, logger=logging):
    return get_embeding_model(
        ctx=ctx,
        nn_symbol=cnn_symbol,
        embedding=embedding,
        feature_num=sentence_size,
        batch_size=batch_size,
        return_grad=return_grad,
        checkpoint=checkpoint,
        logger=logger,
    )


def get_text_cnn_model_without_embedding(ctx, cnn_symbol, vec_size, sentence_size, batch_size, return_grad=False,
                                         checkpoint=None, logger=logging):
    return get_model(
        ctx=ctx,
        nn_symbol=cnn_symbol,
        vec_size=vec_size,
        feature_num=sentence_size,
        batch_size=batch_size,
        return_grad=return_grad,
        checkpoint=checkpoint,
        logger=logger,
    )

def get_keyword_maps(grad_map, sentences, map_range=None, center_tag=False, two_dir_tag=False, tail=0, empty_shift=0,
                     valid_pad=0):
    '''
    梯度转换映射函数
    输入data的梯度值，通过一系列变换，得到每一个单词所对应的关键值
    输入的梯度值的样子
    [
    sentence1: [
        word1: [f1_g], [f2_g], ..., [f_vec_size_g]
        word2: [f1_g], [f2_g], ..., [f_vec_size_g]
        ...
        word_sentence_size: [f1_g], [f2_g], ..., [f_vec_size_g]]
    sentence2: [
        word1: [f1_g], [f2_g], ..., [f_vec_size_g]
        word2: [f1_g], [f2_g], ..., [f_vec_size_g]
        ...
        word_sentence_size: [f1_g], [f2_g], ..., [f_vec_size_g]]
    ...
    sentence_batch_size: [
        word1: [f1_g], [f2_g], ..., [f_vec_size_g]
        word2: [f1_g], [f2_g], ..., [f_vec_size_g]
        ...
        word_sentence_size: [f1_g], [f2_g], ..., [f_vec_size_g]]
    ]
    设定map_range使得每个单词的关键值落在map_range指定的范围内
    设定centen_tag使得如果梯度值映射
    :param grad_map: all gradient of input data
    :param sentences: the sentences to analyse
    :param map_range: default: None, map gradient value to the this range
    :param center_tag: whether or not
    :param two_dir_tag:
    :param tail:
    :param empty_shift:
    :param valid_pad:
    :return:
    '''
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
                        pos_mul_norm, neg_mul_norm = float(max(map_range)) / pos_norm, float(
                            max(map_range)) / neg_norm
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
                    valid_index = [vi + len(sentence) for vi, value in
                                   enumerate(keyword_map[len(sentence): max_extra_valid_idx + 1]) if
                                   abs(value) >= threshold]
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
