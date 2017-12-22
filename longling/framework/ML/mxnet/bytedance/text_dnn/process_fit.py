#coding:utf-8

import sys
sys.path.insert(0,"/opt/tiger/text_lib/")
import os
import json
import logging
import gpu_mxnet as mx

#import mxnet as mx
import numpy as np
from text_iterator import get_w2v,TextIdIterator, get_vocab
# from text_dnn import get_text_dnn_symbol, get_text_dnn_model
from text_dnn import get_text_embedding_dnn_symbol, get_text_embedding_dnn_model
from text_dnn_fit import fit


# def process_fit(location_vec,location_ins,location_test,model_dir,gpu,prop={},epoch=20):
#     batch_size = prop.get('batch_size', 128)
#     num_label = prop.get('num_label', 2)
#     num_hiddens = prop.get('num_hiddens', [100])
#     dropout = prop.get('dropout', 0.5)
#     sentence_size = prop.get('sentence_size', 25)
    
#     w2v = get_w2v(location_vec)
#     print 'w2v', len(w2v)
    
#     text_iter = TextIterator(w2v,location_ins, sentence_size, batch_size)
#     test_iter = TextIterator(w2v,location_test, sentence_size, batch_size)
#     print 'data',text_iter.cnt,test_iter.cnt

#     vec_size = text_iter.vec_size
#     dnn = get_text_dnn_symbol(num_hiddens=num_hiddens, num_label=num_label, dropout=dropout)
#     if int(gpu) >= 0:
#         ctx = mx.gpu(gpu)
#     else:
#         ctx = mx.cpu(0)
#     cm = get_text_dnn_model(ctx, dnn, vec_size, batch_size)
#     print 'batch_size',batch_size
#     print 'vec_size',vec_size
#     location_size = model_dir+'size.txt'
#     s = {}
#     s['sentence_size'] = sentence_size
#     s['batch_size'] = batch_size
#     s['vec_size'] = vec_size
#     s['location_vec'] = location_vec
#     s['epoch_num'] = epoch
#     line = json.dumps(s)
#     with open(location_size,mode='w') as f:
#         f.write(line)
#     fit(cm, text_iter, test_iter, batch_size, epoch=epoch, root=model_dir)
#     text_iter.close()
#     test_iter.close()

def process_fit(location_vec,location_ins,location_test,model_dir,gpu,prop={},epoch=20):
    batch_size = prop.get('batch_size', 128)
    num_label = prop.get('num_label', 2)
    num_hiddens = prop.get('num_hiddens', [100])
    dropout = prop.get('dropout', 0.5)
    sentence_size = prop.get('sentence_size', 25)
    
    # w2v = get_w2v(location_vec)
    id2v, w2id = get_vocab(location_vec) # DONE: id2v is a numpy array
    print 'embedding vocab_size', len(id2v)
    
    text_iter = TextIdIterator(w2id, location_ins, sentence_size) 
    test_iter = TextIdIterator(w2id, location_test, sentence_size)
    print 'data',text_iter.cnt,test_iter.cnt

    vocab_size, vec_size = id2v.shape
    dnn = get_text_embedding_dnn_symbol(vocab_size, vec_size, num_hiddens=num_hiddens, num_label=num_label, dropout=dropout)
    if int(gpu) >= 0:
        ctx = mx.gpu(gpu)
    else:
        ctx = mx.cpu(0)

    checkpoint = None
    if 'checkpoint' in prop:
        checkpoint = prop['checkpoint']
        if not isinstance(checkpoint, int):
            checkpoint = int(checkpoint)
        checkpoint = os.path.join(model_dir, '%s-%04d.params' % ('dnn', checkpoint))
        if not os.path.exists(checkpoint):
            logging.error(u'[text_dnn.process_fit.process_fit] 指定的checkpoint无效！')
            raise ValueError(u'[text_dnn.process_fit.process_fit] 指定的checkpoint无效！')

    cm = get_text_embedding_dnn_model(ctx, dnn, id2v, sentence_size, batch_size, checkpoint)
    print 'batch_size',batch_size
    print 'vec_size',vec_size
    location_size = os.path.join(model_dir, 'size.txt')
    s = {}
    s['sentence_size'] = sentence_size
    s['batch_size'] = batch_size
    s['vec_size'] = vec_size
    s['location_vec'] = location_vec
    s['epoch_num'] = epoch
    s['vocab_size'] = vocab_size
    line = json.dumps(s)
    with open(location_size,mode='w') as f:
        f.write(line)
    fit(cm, text_iter, test_iter, batch_size, epoch=epoch, root=model_dir)
    text_iter.close()
    test_iter.close()

