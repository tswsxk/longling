# uncompyle6 version 2.9.9
# Python bytecode 2.7 (62211)
# Decompiled from: Python 2.7.9 (default, Mar  1 2015, 12:57:24) 
# [GCC 4.9.2]
# Embedded file name: /data01/data/baotengfei/text_cnn4/data_store.py
# Compiled at: 2017-01-11 17:38:41
import json
import numpy as np
import re
import itertools
from collections import Counter

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
            if num !=0 and len(xs) != num:
                #print line
                continue
            w2id[xs[0]] = idx
            idx += 1
            id2v.append(map(float,xs[1:]))
    return np.array(id2v), w2id
    
def pad_sentences(sentences, sentence_size):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    padding_word = '</s>'
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


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [ x[0] for x in word_counts.most_common() ]
    vocabulary = {x:i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([ [ vocabulary[word] for word in sentence ] for sentence in sentences ])
    y = np.array(labels)
    return [x, y]


def get_raw_data(location):
    sentence = []
    label = []
    cnt = 0
    with open(location) as f:
        for line in f:
            cnt += 1
            if cnt % 100000 == 0:
                print cnt
            try:
                n = json.loads(line)
                z = n['z']
                x = n['x']
                x = x.lower()
            except:
                continue

            z = int(z)
            x = x.split()
            label.append(z)
            sentence.append(x)

    return (sentence, label)


def get_data(location, sentence_size):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    sentences, labels = get_raw_data(location)
    sentences_padded = pad_sentences(sentences, sentence_size)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [
     x, y, vocabulary, vocabulary_inv]


def build_input_data_w2v(sentences, labels, w2v):
    """Map sentences and labels to vectors based on a pretrained word2vec"""
    x_vec = []
    for sent in sentences:
        vec = []
        for word in sent:
            if word in w2v:
                vec.append(w2v[word])
            else:
                vec.append(w2v['</s>'])

        x_vec.append(vec)

    x_vec = np.array(x_vec)
    y_vec = np.array(labels)
    print y_vec
    return [
     x_vec, y_vec]


def get_data_w2v(location, w2v, sentence_size):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    sentences, labels = get_raw_data(location)
    sentences_padded = pad_sentences(sentences, sentence_size)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data_w2v(sentences_padded, labels, w2v)
    return [x, y, vocabulary, vocabulary_inv]


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
                continue
            w2v[xs[0]] = map(float, xs[1:])

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
            if num !=0 and len(xs) != num:
                #print line
                continue
            w2id[xs[0]] = idx
            idx += 1
            id2v.append(map(float,xs[1:]))
    return np.array(id2v), w2id

if __name__ == '__main__':
    print 'x'
# okay decompiling data_store.pyc
