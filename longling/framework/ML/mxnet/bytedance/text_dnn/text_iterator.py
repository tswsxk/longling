#coding:utf-8

import json
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
            if num !=0 and len(xs) != num:
                #print line
                continue
            w2v[xs[0]] = np.array(map(float,xs[1:]))
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

def pad_sentences(sentences, sequence_length):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    padding_word = '</s>'
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

# def build_input_data_w2v(sentences, labels, w2v, sentence_size):
#     """Map sentences and labels to vectors based on a pretrained word2vec"""
#     zero = np.zeros_like(w2v['</s>'])
#     # for i in xrange(len(sentences)):
#     #     max_len = max(max_len, len(sentences[i]))
#     sentences = pad_sentences(sentences, sentence_size)
#     x_vec = []
#     for sent in sentences:
#         vec = []   
#         for word in sent:
#             if word in w2v:
#                 vec.append(w2v[word])
#             else:
#                 vec.append(zero)
#         # if len(vec) == 0:
#         #     vec = [w2v['</s>']]
#         # vec = sum(vec)
#         x_vec.append(vec)

#     x_vec = np.array(x_vec)
#     x_vec = np.sum(x_vec, axis=1)
#     y_vec = np.array(labels)
#     return [x_vec, y_vec]


def build_input_data_w2id(sentences, labels, w2id, sentence_size):
    """Map sentences and labels to vectors based on a pretrained word2vec"""
    # zero = np.zeros_like(w2v['</s>'])
    # for i in xrange(len(sentences)):
    #     max_len = max(max_len, len(sentences[i]))
    sentences = pad_sentences(sentences, sentence_size)
    x_vec = []
    for sent in sentences:
        vec = []   
        for word in sent:
            if word in w2id:
                vec.append(w2id[word])
            else:
                vec.append(w2id['</s>'])
        # if len(vec) == 0:
        #     vec = [w2v['</s>']]
        # vec = sum(vec)
        x_vec.append(vec)

    x_vec = np.array(x_vec)
    y_vec = np.array(labels)
    return [x_vec, y_vec]


# class TextIterator(object):
    
#     def __init__(self,w2v, location, sentence_size):
#         self.file = open(location)
#         self.cnt = 0
#         for _ in self.file:
#             self.cnt += 1

#         self.cur = 0

#         self.file.seek(0)
#         self.w2v = w2v
#         vec_size = 0
#         for _,vec in self.w2v.items():
#             vec_size = len(vec)
#             break
#         self.vec_size = vec_size
#         self.sentence_size = sentence_size

#     def close(self):
#         self.file.close()

#     def next_batch(self,batch_size):    
#         xs = []
#         zs = []
#         while len(xs) < batch_size:
#             if self.cur == self.cnt:
#                 self.file.seek(0)
#                 self.cur = 0

#             line = self.file.readline()
#             self.cur += 1
            
#             r = json.loads(line)
#             x = r['x']
#             x = x.split()
#             z = r['z']
#             xs.append(x)
#             zs.append(z)
        
#         # xs = pad_sentences(xs,self.text_size)
#         xs,zs = build_input_data_w2v(xs, zs, self.w2v, self.sentence_size)
#         # xs = np.reshape(xs, (xs.shape[0], 1, xs.shape[1], xs.shape[2]))
#         return xs,zs

class TextIdIterator(object):
    
    # def __init__(self, w2id, location, sentence_size):
        # self.file = open(location)
        # self.cnt = 0
        # for _ in self.file:
        #     self.cnt += 1

        # self.cur = 0

        # self.file.seek(0)
        # self.w2v = w2v

    def __init__(self, w2id, location, sentence_size):
        # self.w2id = w2id
        self.sentence_size = sentence_size
        self.data_buffer = []
        self.label_buffer = []
        self.cnt = 0
        self.pad_id = w2id['</s>']
        with open(location) as fin:
            for line in fin:
                r = json.loads(line)
                self.data_buffer.append([w2id[w] for w in r['x'].split() if w in w2id][:self.sentence_size])
                self.label_buffer.append(r['z'])
                self.cnt += 1
        self.index = 0
    def close(self):
        # self.file.close()
        pass

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

    # def next_batch(self,batch_size):    
    #     xs = []
    #     zs = []
    #     while len(xs) < batch_size:
    #         if self.cur == self.cnt:
    #             self.file.seek(0)
    #             self.cur = 0

    #         line = self.file.readline()
    #         self.cur += 1
            
    #         r = json.loads(line)
    #         x = r['x']
    #         x = x.split()
    #         z = r['z']
    #         xs.append(x)
    #         zs.append(z)
        
    #     # xs = pad_sentences(xs,self.text_size)
    #     xs,zs = build_input_data_w2id(xs, zs, self.w2id, self.sentence_size)
    #     # xs = np.reshape(xs, (xs.shape[0], 1, xs.shape[1], xs.shape[2]))
    #     return xs,zs

# 预读入到buffer中
# import time
# class TextIterator(object):
    
#     def __init__(self,w2v, location, sentence_size, batch_size):
#         self.file = open(location)
#         self.cnt = 0
#         for _ in self.file:
#             self.cnt += 1

#         self.cur = 0

#         self.file.seek(0)
#         self.w2v = w2v
#         vec_size = 0
#         for _,vec in self.w2v.items():
#             vec_size = len(vec)
#             break
#         self.vec_size = vec_size
#         self.sentence_size = sentence_size
#         self.batch_size = batch_size
#         self.buffer_size = self.batch_size * 10
#         self.data_buffer, self.label_buffer = None, None
#         self.update_buffer()
#         self.buffer_ptr = 0


#     def close(self):
#         self.file.close()


#     def update_buffer(self):
#         xs = []
#         zs = []
#         for _ in xrange(self.buffer_size):
#             if self.cur == self.cnt:
#                 self.file.seek(0)
#                 self.cur = 0

#             line = self.file.readline()
#             self.cur += 1
            
#             r = json.loads(line)
#             x = r['x']
#             x = x.split()
#             z = r['z']
#             xs.append(x)
#             zs.append(z)
#         self.data_buffer, self.label_buffer = build_input_data_w2v(xs, zs, self.w2v, self.sentence_size)


#     def next_batch(self):    
#         if self.buffer_ptr == self.buffer_size:
#             self.update_buffer()
#             self.buffer_ptr = 0

#         start_ptr = self.buffer_ptr
#         self.buffer_ptr += self.batch_size
#         return self.data_buffer[start_ptr:self.buffer_ptr], self.label_buffer[start_ptr:self.buffer_ptr]

if __name__ == '__main__':

    print 'x'


    m_vec = get_w2v('vec.s') 
    print 'm_vec',len(m_vec)

    it = TextIter(m_vec,'bait_title.ins',20)

    print it.cnt
    xs,ys = it.next_batch(10)
    print xs
    print ys

    it.close()

