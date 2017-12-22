#coding:utf-8

import sys
sys.path.insert(0,"/opt/tiger/text_lib/")

import gpu_mxnet as mx

import os
#import mxnet as mx
import numpy as np
import time
import math
import json
from collections import namedtuple
# import data_store
from evaluate import evaluate
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logs = sys.stderr
DNNModel = namedtuple('DNNModel', ['dnn_exec', 'symbol', 'data', 'label', 'param_blocks'])

def get_text_embedding_dnn_symbol(vocab_size, embed_dim, num_hiddens=[100], num_label=2, dropout=0.0):
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')
    net = mx.sym.Embedding(data=data, input_dim=vocab_size, output_dim=embed_dim, name='embedding')
    net = mx.sym.sum(net, axis=1, name="sum")
   
    for i in xrange(len(num_hiddens)):
        net = mx.sym.FullyConnected(data=net, name='fc%s' % i, num_hidden=num_hiddens[i])
        net = mx.sym.Activation(data=net, name='relu%s' % i, act_type="relu")
    if dropout > 0.0:
        net = mx.sym.Dropout(data=net, p=dropout)
    net = mx.sym.FullyConnected(data=net, name='fc', num_hidden=num_label)
    softmax_label = mx.sym.SoftmaxOutput(data=net, label=label, name='softmax')
    return softmax_label

# def get_text_dnn_symbol(num_hiddens=[100], num_label=2, dropout=0.0):
#     data = mx.sym.Variable('data')
#     label = mx.sym.Variable('softmax_label')
#     layers = [data]
#     for i in xrange(len(num_hiddens)):
#         z = mx.sym.FullyConnected(data=layers[-1], name='fc%s' % i, num_hidden=num_hiddens[i])
#         a = mx.sym.Activation(data=z, name='relu%s' % i, act_type="relu")
#         layers.append(a)
#     if dropout > 0.0:
#         layers.append(mx.sym.Dropout(data=layers[-1], p=dropout))
#     output = mx.sym.FullyConnected(data=layers[-1], name='fc', num_hidden=num_label)
#     softmax_label = mx.sym.SoftmaxOutput(data=output, label=label, name='softmax')
#     return softmax_label

def load_text_dnn_symbol(location, batch_size=128, is_train=False):
    with open(location) as f:
        text = f.read()
    x = json.loads(text)
    for n in x['nodes']:
        if n['op'] == 'Dropout' and not is_train:
            n['param']['p'] = "0.0"
    text = json.dumps(x)
    symbol = mx.symbol.load_json(text)
    return symbol

# added
# def get_text_embedding_dnn_model(ctx, dnn, embedding, max_len, batch_size, initializer=mx.initializer.Uniform(0.1)):
#     arg_names = dnn.list_arguments()
#     input_shapes = {}
#     input_shapes['data'] = (batch_size, max_len)
#     arg_shape, out_shape, aux_shape = dnn.infer_shape(**input_shapes)
    
#     arg_arrays = [ mx.nd.zeros(s, ctx) for s in arg_shape ]
#     args_grad = {}
#     for shape, name in zip(arg_shape, arg_names):
#         if name in ('softmax_label', 'data', 'embedding_weight'):
#             continue
#         args_grad[name] = mx.nd.zeros(shape, ctx)

#     dnn_exec = dnn.bind(ctx=ctx, args=arg_arrays, args_grad=args_grad, grad_req='add')
#     param_blocks = []
#     arg_dict = dict(zip(arg_names, dnn_exec.arg_arrays))
#     for i, name in enumerate(arg_names):
#         if name in ('softmax_label', 'data'):
#             continue
#         if name == 'embedding_weight':
#             arg_dict[name][:] = embedding   # 用预训练的词向量初始化 embedding layer， 模型不存储embedding
#             continue 
#         initializer(name, arg_dict[name])
#         param_blocks.append((i, arg_dict[name], args_grad[name], name))

#     out_dict = dict(zip(dnn.list_outputs(), dnn_exec.outputs))
#     data = dnn_exec.arg_dict['data']
#     label = dnn_exec.arg_dict['softmax_label']
#     return DNNModel(dnn_exec=dnn_exec, symbol=dnn, data=data, label=label, param_blocks=param_blocks)


def get_text_embedding_dnn_model(ctx, dnn, embedding, sentence_size, batch_size, checkpoint=None):
    arg_names = dnn.list_arguments()
    input_shapes = {}
    input_shapes['data'] = (batch_size, sentence_size)
    arg_shape, out_shape, aux_shape = dnn.infer_shape(**input_shapes)

    # init space for grads
    args_grad = {}
    for name, shape in zip(arg_names, arg_shape):
        if name in ('softmax_label', 'data', 'embedding_weight'):
            continue
        args_grad[name] = mx.nd.zeros(shape, ctx)

    arg_dict = {}
    if checkpoint != None:
        print "checkpoint", checkpoint
        p = mx.nd.load(checkpoint)
        for k, v in p.items():
            if k.startswith('arg:'):
                k = k[4:]
                arg_dict[k] = mx.nd.array(v.asnumpy(), ctx)
    initializer=mx.initializer.Uniform(0.1)
    arg_arrays = []
    param_blocks = []

    for i, (name, shape) in enumerate(zip(arg_names, arg_shape)):
        if name in ('softmax_label', 'data'):
            arg_arrays.append(mx.nd.zeros(shape, ctx))
            continue
        if name == "embedding_weight":
            arg_arrays.append(mx.nd.array(embedding, ctx))
            continue
        if name not in arg_dict:
            arg_dict[name] = mx.nd.zeros(shape, ctx)
            initializer(name, arg_dict[name])
        arg_arrays.append(arg_dict[name])
        param_blocks.append((i, arg_dict[name], args_grad[name], name))
    print "finish loading"
    dnn_exec = dnn.bind(ctx=ctx, args=arg_arrays, args_grad=args_grad, grad_req='add')

    out_dict = dict(zip(dnn.list_outputs(), dnn_exec.outputs))
    data = dnn_exec.arg_dict['data']
    label = dnn_exec.arg_dict['softmax_label']
    return DNNModel(dnn_exec=dnn_exec, symbol=dnn, data=data, label=label, param_blocks=param_blocks)

# def get_text_dnn_model(ctx, dnn, vec_size, batch_size, initializer=mx.initializer.Uniform(0.1)):
#     arg_names = dnn.list_arguments()
#     input_shapes = {}
#     input_shapes['data'] = (batch_size, vec_size)
#     arg_shape, out_shape, aux_shape = dnn.infer_shape(**input_shapes)
#     arg_arrays = [ mx.nd.zeros(s, ctx) for s in arg_shape ]
#     args_grad = {}
#     for shape, name in zip(arg_shape, arg_names):
#         if name in ('softmax_label', 'data'):
#             continue
#         args_grad[name] = mx.nd.zeros(shape, ctx)

#     dnn_exec = dnn.bind(ctx=ctx, args=arg_arrays, args_grad=args_grad, grad_req='add')
#     param_blocks = []
#     arg_dict = dict(zip(arg_names, dnn_exec.arg_arrays))
#     for i, name in enumerate(arg_names):
#         if name in ('softmax_label', 'data'):
#             continue
#         if name == 'embedding':
#             arg_dict[name][:] = embedding   # 用预训练的词向量初始化 embedding layer， 模型不存储embedding
#             continue 
#         initializer(name, arg_dict[name])
#         param_blocks.append((i, arg_dict[name], args_grad[name], name))

#     out_dict = dict(zip(dnn.list_outputs(), dnn_exec.outputs))
#     data = dnn_exec.arg_dict['data']
#     label = dnn_exec.arg_dict['softmax_label']
#     return DNNModel(dnn_exec=dnn_exec, symbol=dnn, data=data, label=label, param_blocks=param_blocks)

# def learn(model, X_train_batch, y_train_batch, X_dev_batch, y_dev_batch, batch_size, optimizer='rmsprop', max_grad_norm=5.0, learning_rate=0.0005, epoch=200, root='model/'):
#     m = model
#     opt = mx.optimizer.create(optimizer)
#     opt.lr = learning_rate
#     updater = mx.optimizer.get_updater(opt)
#     location_res = root+'model.res'
#     print location_res
#     f_res = open(location_res,'w')
#     for iteration in range(epoch):
#         tic = time.time()
#         num_correct = 0
#         num_total = 0
#         for begin in range(0, X_train_batch.shape[0], batch_size):
#             batchX = X_train_batch[begin:begin + batch_size]
#             batchY = y_train_batch[begin:begin + batch_size]
#             #print 'x',batchX.shape,batchY.shape
#             if batchX.shape[0] != batch_size:
#                 continue
#             m.data[:] = batchX
#             m.label[:] = batchY
#             m.dnn_exec.forward(is_train=True)
#             m.dnn_exec.backward()
#             num_correct += sum(batchY == np.argmax(m.dnn_exec.outputs[0].asnumpy(), axis=1))
#             num_total += len(batchY)
#             norm = 0
#             for idx, weight, grad, name in m.param_blocks:
#                 grad /= batch_size
#                 l2_norm = mx.nd.norm(grad).asscalar()
#                 norm += l2_norm * l2_norm

#             norm = math.sqrt(norm)
#             for idx, weight, grad, name in m.param_blocks:
#                 if norm > max_grad_norm:
#                     grad *= max_grad_norm / norm
#                 updater(idx, grad, weight)
#                 grad[:] = 0.0

#         if iteration % 50 == 0 and iteration > 0:
#             opt.lr *= 0.5
#             print >> logs, 'reset learning rate to %g' % opt.lr
#         toc = time.time()
#         train_time = toc - tic
#         train_acc = num_correct * 100 / float(num_total)
#         if (iteration + 1) % 1 == 0:
#             prefix = root + 'dnn'
#             m.symbol.save('%s-symbol.json' % prefix)
#             save_dict = {'arg:%s' % k:v for k, v in m.dnn_exec.arg_dict.items()}
#             save_dict.update({'aux:%s' % k:v for k, v in m.dnn_exec.aux_dict.items()})
#             param_name = '%s-%04d.params' % (prefix, iteration)
#             mx.nd.save(param_name, save_dict)
#             print >> logs, 'Saved checkpoint to %s' % param_name
#         num_correct = 0
#         num_total = 0
#         ps = []
#         for begin in range(0, X_dev_batch.shape[0], batch_size):
#             batchX = X_dev_batch[begin:begin + batch_size]
#             batchY = y_dev_batch[begin:begin + batch_size]
#             if batchX.shape[0] != batch_size:
#                 continue
#             m.data[:] = batchX
#             m.dnn_exec.forward(is_train=False)
#             ps.extend(np.argmax(m.dnn_exec.outputs[0].asnumpy(), axis=1))
#             num_correct += sum(batchY == np.argmax(m.dnn_exec.outputs[0].asnumpy(), axis=1))
#             num_total += len(batchY)

#         cat_res = evaluate(y_dev_batch, ps)
#         dev_acc = num_correct * 100 / float(num_total)
#         print >> logs, 'Iter [%d] Train: Time: %.3fs, Training Accuracy: %.3f' % (iteration, train_time, train_acc)
#         print >> logs, '--- Dev Accuracy thus far: %.3f' % dev_acc

#         line = 'Iter [%d] Train: Time: %.3fs, Training Accuracy: %.3f' % (iteration, train_time, train_acc)
#         f_res.write(line.encode('utf-8')+'\n')
#         f_res.flush()

#         line = '--- Dev Accuracy thus far: %.3f' % dev_acc
#         f_res.write(line.encode('utf-8')+'\n')
#         f_res.flush()

#         for cat, res in cat_res.items():
#             print >> logs, '--- Dev Category %s P=%s,R=%s,F=%s' % (cat, res[0], res[1], res[2])

#             line = '--- Dev Category %s P=%s,R=%s,F=%s' % (cat, res[0], res[1], res[2])
#             f_res.write(line.encode('utf-8')+'\n')
#             f_res.flush()
#     f_res.close()
