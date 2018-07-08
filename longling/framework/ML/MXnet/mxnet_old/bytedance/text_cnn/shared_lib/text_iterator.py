# coding:utf-8

import json
import numpy as np

class TextIterator(object):
    def __init__(self, w2id, location, sentence_size):
        self.cnt = 0
        self.data_buffer = []
        self.label_buffer = []
        self.sentence_size = sentence_size
        with open(location) as fin:
            for line in fin:
                data = json.loads(line, encoding='utf8')
                self.data_buffer.append([w2id[w] for w in data['x'].split() if w in w2id][:sentence_size])
                self.label_buffer.append(data['z'])
                self.cnt += 1
        self.index = 0
        self.pad_id = w2id['</s>']

    def pad(self, sentences):
        for i in xrange(len(sentences)):
            len_ = len(sentences[i])
            sentences[i] += [self.pad_id] * (self.sentence_size - len_)

    def next_batch(self, batch_size):
        data_batch = []
        label_batch = []
        for _ in xrange(batch_size):
            data_batch.append(self.data_buffer[self.index])
            label_batch.append(self.label_buffer[self.index])
            self.index = (self.index + 1) % self.cnt
        self.pad(data_batch)
        return np.array(data_batch), np.array(label_batch)



