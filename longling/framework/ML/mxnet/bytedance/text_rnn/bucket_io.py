# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.insert(0, "/opt/tiger/nlp/text_env/env/lib/python2.7/site-packages/mxnet-0.9.5-py2.7.egg")
import mxnet as mx
import json

from utils import to_unicode


def default_read_content(path):
    sentences = []
    sent2label = {}
    with open(path) as ins:
        for line in ins:
            d = json.loads(to_unicode(line))
            sent = d['x']
            label = d['z']
            sentences.append(sent)
            sent2label[sent] = int(label)
    return sentences, sent2label

def default_read_content_score(path):
    ids = []
    id2rec = {}
    with open(path) as ins:
        for line in ins:
            rec = json.loads(to_unicode(line))
            ids.append(rec['id'])
            id2rec[rec['id']] = rec
    return ids, id2rec


def default_text2id(sentence, the_vocab):
    #words = list(to_unicode(sentence)) #char
    words = to_unicode(sentence).split() #word
    words = [the_vocab.get(w, 0) for w in words if len(w) > 0]
    words = [k for k in words if k!=0]
    return words

def default_conv_vocab(vocab):
    vocab2id = {}
    id2vec = {}
    i = 1
    for k,v in vocab:#[[w,v],[w,v]]
        k = to_unicode(k)
        vocab2id[k] = i
        id2vec[i] = v
        i += 1
    id2vec[0] = [0]*len(v)
    num_embed = len(v)
    return vocab2id,id2vec,num_embed

def default_gen_buckets(sentences, batch_size, the_vocab, text2id):
    len_dict = {}
    max_len = -1
    for sentence in sentences:
        words = text2id(sentence, the_vocab)
        if len(words) == 0:
            continue
        if len(words) > max_len:
            max_len = len(words)
        if len(words) in len_dict:
            len_dict[len(words)] += 1
        else:
            len_dict[len(words)] = 1

    tl = 0
    buckets = []
    for l, n in len_dict.items(): # TODO: There are better heuristic ways to do this
        if n + tl >= batch_size:
            buckets.append(l)
            tl = 0
        else:
            tl += n
    if tl > 0:
        buckets.append(max_len)
    return buckets


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class ScoreBatch(object):
    def __init__(self, data_names, data, label_names, label, rec, bucket_key, length):
        self.data = data
        self.rec = rec
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

        self.pad = 0
        self.index = None # TODO: what is index?
        self.length = length

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class DummyIter(mx.io.DataIter):
    "A dummy iterator that always return the same batch, used for speed testing"
    def __init__(self, real_iter):
        super(DummyIter, self).__init__()
        self.real_iter = real_iter
        self.provide_data = real_iter.provide_data
        self.provide_label = real_iter.provide_label
        self.batch_size = real_iter.batch_size

        for batch in real_iter:
            self.the_batch = batch
            break

    def __iter__(self):
        return self

    def next(self):
        return self.the_batch


class BucketSentenceIter(mx.io.DataIter):
    def __init__(self, path, vocab, buckets, batch_size,
                 init_states, data_name='data', label_name='label',
                 seperate_char=' <eos> ', text2id=None, read_content=None):
        super(BucketSentenceIter, self).__init__()

        vocab2id,self.id2vec,self.num_embed = default_conv_vocab(vocab)

        if text2id == None:
            self.text2id = default_text2id
        else:
            self.text2id = text2id
        if read_content == None:
            self.read_content = default_read_content
        else:
            self.read_content = read_content

        sentences, self.sent2label = self.read_content(path)

        if len(buckets) == 0:
            buckets = default_gen_buckets(sentences, batch_size, vocab2id, self.text2id)

        self.vocab_size = len(vocab2id)
        self.data_name = data_name
        self.label_name = label_name

        buckets.sort()
        self.buckets = buckets
        self.data = [[] for _ in buckets]
        self.data_label = [[] for _ in buckets]
        # pre-allocate with the largest bucket for better memory sharing
        self.default_bucket_key = max(buckets) * self.num_embed#vec  default_bucket_key干嘛用的

        for sentence in sentences:
            label = self.sent2label[sentence]
            sentence = self.text2id(sentence, vocab2id)
            self.sent2label[tuple([int(x) for x in sentence if x > 0])] = label
            if len(sentence) == 0:
                continue
            for i, bkt in enumerate(buckets):
                if bkt >= len(sentence):
                    self.data[i].append(sentence)
                    break
            # we just ignore the sentence it is longer than the maximum
            # bucket size here

        # convert data into ndarrays for better speed during training
        data = [np.zeros((len(x), buckets[i])) for i, x in enumerate(self.data)] # 每个元素的shape是 [每个bucket中的数据量，长度]
        for i_bucket in range(len(self.buckets)):
            for j in range(len(self.data[i_bucket])):
                sentence = self.data[i_bucket][j]
                data[i_bucket][j, :len(sentence)] = sentence
        self.data = data

        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        bucket_sizes = [len(x) for x in self.data]

        print("Summary of dataset ==================")
        for bkt, size in zip(buckets, bucket_sizes):
            print("bucket of len %3d : %d samples" % (bkt, size)) #fix
        print("total %s useful samples" % sum(bucket_sizes))    
        self.batch_size = batch_size
        self.make_data_iter_plan()

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [('data', (batch_size, self.default_bucket_key))] + init_states
        ## fixme
        ## self.provide_label = [('softmax_label', (self.batch_size, self.default_bucket_key))]
        self.provide_label = [('softmax_label', (self.batch_size, 1))]

    def make_data_iter_plan(self):
        "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        bucket_n_batches = []
        for i in range(len(self.data)):
            bucket_n_batches.append(len(self.data[i]) / self.batch_size)
            self.data[i] = self.data[i][:int(bucket_n_batches[i]*self.batch_size)]

        bucket_plan = np.hstack([np.zeros(n, int)+i for i, n in enumerate(bucket_n_batches)])
        np.random.shuffle(bucket_plan)

        bucket_idx_all = [np.random.permutation(len(x)) for x in self.data]

        self.bucket_plan = bucket_plan
        self.bucket_idx_all = bucket_idx_all
        self.bucket_curr_idx = [0 for x in self.data]

        self.data_buffer = []  # 每种bucket的数据区域 但是这里没有转成vec
        self.label_buffer = []

        for i_bucket in range(len(self.data)):
            data = np.zeros((self.batch_size, self.buckets[i_bucket]))
            ## fixme
            ## label = np.zeros((self.batch_size, self.buckets[i_bucket]))
            label = np.zeros((self.batch_size, 1))
            self.data_buffer.append(data)
            self.label_buffer.append(label)

    def __iter__(self):

        for i_bucket in self.bucket_plan:
            data = self.data_buffer[i_bucket]
            i_idx = self.bucket_curr_idx[i_bucket]
            idx = self.bucket_idx_all[i_bucket][i_idx:i_idx+self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size

            init_state_names = [x[0] for x in self.init_states]
            data[:] = self.data[i_bucket][idx]

            for sentence in data:
                assert len(sentence) == self.buckets[i_bucket]

            ## fixme
            label = self.label_buffer[i_bucket]
            ## label[:, :-1] = data[:, 1:]
            ## label[:, -1] = 0
            #label[:,] = 1  # fixme
            for idx, sentence in enumerate(data):
                key = tuple([int(x) for x in sentence if int(x) > 0])
                _label = self.sent2label[key]
                label[idx][:] = _label

            data2 = []#init embeding
            for idx, sentence in enumerate(data):
                vec = [self.id2vec.get(i,) for i in sentence]
                vec = np.concatenate(vec,axis=0)
                data2.append(vec)
            data = np.array(data2)

            data_all = [mx.nd.array(data)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data'] + init_state_names
            label_names = ['softmax_label']

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
                                     self.buckets[i_bucket]*self.num_embed)#vec
            data_shapes = {}
            #for x in data_batch.provide_data + data_batch.provide_label:
            #    data_shapes[x[0]] = tuple(['slice'] + list(x[1][1:]))
            #print 'provide_data,data_shapes',data_shapes

            yield data_batch


    def reset(self):
        self.bucket_curr_idx = [0 for x in self.data]

class BucketSentenceScoreIter(mx.io.DataIter):
    def __init__(self, path, vocab, buckets, batch_size,
                 init_states, data_name='data',
                 seperate_char=' <eos> ', text2id=None, read_content=None):
        super(BucketSentenceScoreIter, self).__init__()

        vocab2id, self.id2vec, self.num_embed = default_conv_vocab(vocab)

        if text2id == None:
            self.text2id = default_text2id
        else:
            self.text2id = text2id
        if read_content == None:
            self.read_content = default_read_content_score
        else:
            self.read_content = read_content

        ids, self.id2rec = self.read_content(path)
        sentences = [x['data'] for x in self.id2rec.values()]

        if len(buckets) == 0:
            buckets = default_gen_buckets(sentences, batch_size, vocab2id, self.text2id)

        self.vocab_size = len(vocab2id)
        self.data_name = data_name

        buckets.sort()
        self.buckets = buckets
        self.ids = [[] for _ in buckets]
        self.data = [[] for _ in buckets]
        self.data_label = [[] for _ in buckets]
        # pre-allocate with the largest bucket for better memory sharing
        self.default_bucket_key = max(buckets) * self.num_embed#vec
        self.zero_len_sentences_id = []
        for the_id in ids:
            rec = self.id2rec[the_id]
            sentence = self.text2id(rec['data'], vocab2id)
            if len(sentence) == 0:
                self.zero_len_sentences_id.append(the_id)
                continue
            if len(sentence) > buckets[-1]:
                self.data[-1].append(sentence[:buckets[-1]])
                self.ids[-1].append(the_id)
                continue
            for i, bkt in enumerate(buckets):
                if bkt >= len(sentence):
                    self.data[i].append(sentence)
                    self.ids[i].append(the_id)
                    break

        # convert data into ndarrays for better speed during training
        data = [np.zeros((len(x), buckets[i])) for i, x in enumerate(self.data)]
        for i_bucket in range(len(self.buckets)):
            for j in range(len(self.data[i_bucket])):
                sentence = self.data[i_bucket][j]
                data[i_bucket][j, :len(sentence)] = sentence
        self.data = data

        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        bucket_sizes = [len(x) for x in self.data]

        print("Summary of dataset ==================")
        for bkt, size in zip(buckets, bucket_sizes):
            print("bucket of len %3d : %d samples" % (bkt, size)) #fix
        print "total %s samples" % len(ids)
        print "total %s useful samples" % sum(bucket_sizes)
        print "%s zero-len samples" % len(self.zero_len_sentences_id)
        self.batch_size = batch_size
        self.make_data_iter_plan()

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [('data', (batch_size, self.default_bucket_key))] + init_states
        self.provide_label = [('softmax_label', (self.batch_size, 1))]

    def make_data_iter_plan(self):
        "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        bucket_n_batches = []
        for i in range(len(self.data)):
            bucket_n_batches.append((len(self.data[i]) + self.batch_size - 1) / self.batch_size)
            # self.data[i] = self.data[i][:int(bucket_n_batches[i]*self.batch_size)]

        bucket_plan = np.hstack([np.zeros(n, int)+i for i, n in enumerate(bucket_n_batches)])

        bucket_idx_all = [np.array(range(len(x))) for x in self.data]

        self.bucket_plan = bucket_plan
        self.bucket_idx_all = bucket_idx_all
        self.bucket_curr_idx = [0 for x in self.data]

        self.data_buffer = []
        self.rec_buffer = []
        self.label_buffer = []
        for i_bucket in range(len(self.data)):
            data = np.zeros((self.batch_size, self.buckets[i_bucket]))
            ## fixme
            ## label = np.zeros((self.batch_size, self.buckets[i_bucket]))
            rec = [{} for _ in range(self.batch_size)]
            label = np.zeros((self.batch_size, 1))
            self.data_buffer.append(data)
            self.rec_buffer.append(rec)
            self.label_buffer.append(label)

    def __iter__(self):
        for i_bucket in self.bucket_plan:
            data = self.data_buffer[i_bucket]
            i_idx = self.bucket_curr_idx[i_bucket]
            if i_idx + self.batch_size <= len(self.bucket_idx_all[i_bucket]):
                idx = self.bucket_idx_all[i_bucket][i_idx:(i_idx + self.batch_size)]
            else:
                idx = self.bucket_idx_all[i_bucket][i_idx:]
            idx_len = len(idx)
            self.bucket_curr_idx[i_bucket] += self.batch_size

            init_state_names = [x[0] for x in self.init_states]
            data[:len(idx)] = self.data[i_bucket][idx]
            ids = [self.ids[i_bucket][x] for x in idx]

            for sentence in data:
                assert len(sentence) == self.buckets[i_bucket]

            label = self.label_buffer[i_bucket]
            rec = self.rec_buffer[i_bucket]
            for idx in range(idx_len):
                _rec = self.id2rec[ids[idx]]
                rec[idx] = _rec
                label[idx][:] = int(_rec['label']) if _rec['label'].isdigit() else -1

            label_all = [mx.nd.array(label)]
            data2 = []#init embeding
            for idx, sentence in enumerate(data):
                vec = [self.id2vec.get(i,) for i in sentence]
                vec = np.concatenate(vec,axis=0)
                data2.append(vec)
            data = np.array(data2)

            data_all = [mx.nd.array(data)] + self.init_state_arrays
            rec_all = rec
            data_names = ['data'] + init_state_names
            label_names = ['softmax_label']

            data_batch = ScoreBatch(data_names, data_all, label_names, label_all, rec_all,
                                     self.buckets[i_bucket] * self.num_embed, idx_len)#vec

            yield data_batch

    def zero_len_rec(self):
        return [self.id2rec[the_id] for the_id in self.zero_len_sentences_id]

    def reset(self):
        self.bucket_curr_idx = [0 for x in self.data]
