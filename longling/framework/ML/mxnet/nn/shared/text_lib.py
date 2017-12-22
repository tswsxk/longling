# coding: utf-8
# create by tongshiwei on 2017/10/23
from __future__ import unicode_literals

import logging

import numpy as np

from longling.base import unistr
from longling.lib.stream import rf_open

def to_unicode(s):
    return unistr(s) if isinstance(s, str) else s


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


def get_w2v(location):
    w2v = {}
    num = 0
    with rf_open(location) as f:
        for line in f:
            line = unistr(line)
            xs = line.strip().split()
            if len(xs) == 2:
                continue
            if num == 0:
                num = len(xs)
            if num != 0 and len(xs) != num:
                # print line
                continue
            w2v[xs[0]] = map(float, xs[1:])
    return w2v

def conv_w2v(w2v, sentences):
    sentences = [sentence.split() for sentence in sentences]
    sentences_padded = pad_sentences(sentences)
    x_vec = []
    for sent in sentences_padded:
        vec = []
        for word in sent:
            if word in w2v:
                vec.append(w2v[word])
            else:
                vec.append(w2v['</s>'])
        x_vec.append(vec)
    x_vec = np.array(x_vec)

    return x_vec

def conv_w2id(w2id, sentences):
    sentences = [sentence.split() for sentence in sentences]
    sentences_padded = pad_sentences(sentences)
    x_vec = []
    for sent in sentences_padded:
        vec = []
        for word in sent:
            if word in w2id:
                vec.append(w2id[word])
            else:
                vec.append(w2id['</s>'])
        x_vec.append(vec)
    x_vec = np.array(x_vec)

    return x_vec


def get_vocab(location):
    id2v = []
    w2id = {}
    idx = 0
    num = 0
    with rf_open(location) as f:
        for line in f:
            line = unistr(line)
            xs = line.strip().split()
            if len(xs) == 2:
                continue
            if num == 0:
                num = len(xs)
            if num != 0 and len(xs) != num:
                continue
            w2id[xs[0]] = idx
            idx += 1
            id2v.append(map(float, xs[1:]))
    return np.array(id2v), w2id


class vecDict():
    def __init__(self, location_vec, logger=logging):
        self.location_vec = location_vec
        logger.info("building vocab")
        self.id2v, self.w2id = get_vocab(location_vec)
        logger.info('w2id-%s' % len(self.w2id))
        self.vocab_size, self.vec_size = self.id2v.shape

    @property
    def info(self):
        return {'location_vec': self.location_vec, 'vocab_size': self.vocab_size, 'vec_size': self.vec_size}

    @property
    def embedding(self):
        return self.id2v