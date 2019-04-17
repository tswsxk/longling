# coding: utf-8
# created by tongshiwei on 18-1-27

from __future__ import unicode_literals

import logging

import numpy as np

from longling.base import unistr
from longling.lib.stream import rf_open

from tqdm import tqdm


def as_list(obj):
    if isinstance(obj, (list, tuple)):
        return obj
    else:
        return [obj]


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


def sentences_preprocess(sentences, padding_word="</s>", sentence_size=None, sentence_split=True):
    if sentence_split:
        sentences = [sentence.split() for sentence in sentences]
    if sentence_size is not None:
        assert isinstance(sentence_size, int)
        sentences = pad_sentences(sentences, padding_word, sentence_size)
    return sentences


def get_w2v(location):
    w2v = {}
    num = 0
    with rf_open(location) as f:
        for line in tqdm(f, desc="reading word2vec dict file[%s]" % location):
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


def get_vocab(location):
    id2v = []
    w2id = {}
    idx = 0
    num = 0
    with rf_open(location) as f:
        for line in tqdm(f, desc="reading word2vec dict file[%s]" % location):
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
            id2v.append(list(map(float, xs[1:])))
    return np.array(id2v), w2id


class VecDict(object):
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

    def w2i(self, words, padding_word="</s>"):
        words = as_list(words)
        vecs = [self.w2id[word] if word in self.w2id else self.w2id[padding_word] for word in words]
        return vecs

    def _w2v(self, word):
        return self.id2v[self.w2id[word]]

    def w2v(self, words, padding_word="</s>"):
        words = as_list(words)
        vecs = [self._w2v(word) if word in self.w2id else self._w2v(padding_word) for word in words]
        return vecs

    def s2i(self, sentences, padding_word="</s>", sentence_size=None, sentence_split=True):
        sentences = sentences_preprocess(sentences, padding_word, sentence_size, sentence_split)
        sentences_ids = []

        for sent in sentences:
            sentences_ids.append(self.w2i(sent, padding_word))
        sentences_ids = np.array(sentences_ids)

        return sentences_ids

    def s2v(self, sentences, padding_word="</s>", sentence_size=None, sentence_split=True):
        sentences = sentences_preprocess(sentences, padding_word, sentence_size, sentence_split)
        sentences_vec = []

        for sent in sentences:
            sentences_vec.append(self.w2v(sent, padding_word))
        sentences_vec = np.array(sentences_vec)

        return sentences_vec


class Word2VecDict(object):
    def __init__(self, location_vec, logger=logging):
        self.location_vec = location_vec
        logger.info("building vocab")
        self.w2v_dict = get_w2v(location_vec)
        logger.info('w2v-%s' % len(self.w2v_dict))
        self.vocab_size, self.vec_size = len(self.w2v_dict), len(self.w2v_dict.values()[0])

    @property
    def info(self):
        return {'location_vec': self.location_vec, 'vocab_size': self.vocab_size, 'vec_size': self.vec_size}

    def w2v(self, words, padding_word="</s>"):
        words = as_list(words)
        vecs = [self.w2v_dict[word] if word in self.w2v_dict else padding_word for word in words]
        return vecs

    def s2v(self, sentences, padding_word="</s>", sentence_size=None, sentence_split=True):
        sentences = sentences_preprocess(sentences, padding_word, sentence_size, sentence_split)
        sentences_vec = []
        for sent in sentences:
            sentences_vec.append(self.w2v(sent, padding_word))
        sentences_vec = np.array(sentences_vec)

        return sentences_vec
