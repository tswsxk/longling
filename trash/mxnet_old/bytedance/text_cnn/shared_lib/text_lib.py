# coding: utf-8

from collections import defaultdict
import numpy as np

def get_w2v(location):
    w2v = {}
    num = 0
    with open(location) as f:
        for line in f:
            line = line.decode('utf-8')
            xs = line.strip().split()
            if len(xs) == 2:
                continue
            if num == 0:
                num = len(xs)
            if num != 0 and len(xs) != num:
                # print line
                continue
            w2v[xs[0]] = map(float, xs[1:])
    # num = len(w2v.itervalues().next())
    return w2v


def get_vocab(location):
    id2v = []
    w2id = {}
    idx = 0
    num = 0
    with open(location) as f:
        for line in f:
            line = line.decode('utf8')
            xs = line.strip().split()
            if len(xs) == 2:
                continue
            if num == 0:
                num = len(xs)
            if num != 0 and len(xs) != num:
                # print line
                continue
            w2id[xs[0]] = idx
            idx += 1
            id2v.append(map(float, xs[1:]))
    return np.array(id2v), w2id


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

def _evaluate(gold_set, predict_set):
    if len(gold_set) == 0:
        print 'gold set is empty'
        return 0, 1, 0
        # raise ValueError('gold set cannot be empty')
    if len(predict_set) == 0:
        return 1, 0, 0
    n_common = len(gold_set & predict_set)
    precision = n_common * 1.0 / len(predict_set)
    recall = n_common * 1.0 / len(gold_set)
    if precision + recall == 0:
        return  precision, recall, 0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def evaluate(gold_list, predict_list):
    cat_gold = defaultdict(set)
    cat_pred = defaultdict(set)
    for i, (g, p) in enumerate(zip(gold_list, predict_list)):
        if g not in cat_gold:
            cat_gold[g] = set()
        if p not in cat_pred:
            cat_pred[p] = set()
        cat_gold[g].add(i)
        cat_pred[p].add(i)


    cat_res = dict()
    for cat in cat_gold.keys():
        p, r, f = _evaluate(cat_gold[cat], cat_pred[cat])
        #print '%s: precision=%s recall=%s f1=%s' % (cat, p, r, f)
        cat_res[cat] = (p, r, f)

    return cat_res