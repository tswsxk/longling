# coding:utf-8

import json
import mxnet as mx
import numpy as np
import time
import math
from collections import namedtuple
import logging

from text_lib import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
CNNModel = namedtuple('CNNModel', ['cnn_exec', 'symbol', 'data', 'label', 'param_blocks'])

def get_text_cnn_symbol_without_loss(sentence_size, vec_size, batch_size, vocab_size=None,
                        num_label=2, filter_list=[1, 2, 3, 4], num_filter=60, dropout=0.0):

    input_x = mx.sym.Variable('data')  # batch_size, sentence_size

    if vocab_size is not None:
        embed_x = mx.sym.Embedding(data=input_x, input_dim=vocab_size, output_dim=vec_size, name="embedding")
        conv_input = mx.sym.Reshape(data=embed_x, target_shape=(batch_size, 1, sentence_size, vec_size))
    else:
        conv_input = input_x

    pooled_outputs = []
    for i, filter_size in enumerate(filter_list):
        convi = mx.sym.Convolution(data=conv_input, kernel=(filter_size, vec_size), num_filter=num_filter,
                                   name=('convolution%s' % i))
        relui = mx.sym.Activation(data=convi, act_type='relu', name=('activation%s' % i))
        pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(sentence_size - filter_size + 1, 1), stride=(1,
                                                                                                                 1),
                               name=('pooling%s' % i))
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

    input_y = mx.sym.Variable('softmax_label')
    fc = get_text_cnn_symbol_without_loss(sentence_size, vec_size, batch_size, vocab_size, num_label, filter_list, num_filter, dropout)
    sm = mx.sym.SoftmaxOutput(data=fc, label=input_y, name='softmax')

    return sm


def load_text_cnn_symbol(location, batch_size=128, is_train=False):
    with open(location) as f:
        text = f.read()
    x = json.loads(text)
    # 改变预测时的batch_size
    for n in x['nodes']:
        if n['op'] == 'Reshape':
            shape = eval(n['param']['target_shape'])
            shape = (batch_size,) + shape[1:]
            n['param']['target_shape'] = '(%s)' % ','.join([str(s) for s in shape])
        if n['op'] == 'Dropout' and not is_train:
            n['param']['p'] = "0.0"

    text = json.dumps(x)
    sm = mx.symbol.load_json(text)
    return sm


def get_text_cnn_model(ctx, cnn, embedding, sentence_size, batch_size, checkpoint=None):
    arg_names = cnn.list_arguments()
    input_shapes = {}
    input_shapes['data'] = (batch_size, sentence_size)
    arg_shape, out_shape, aux_shape = cnn.infer_shape(**input_shapes)

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
    initializer = mx.initializer.Uniform(0.1)
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
    cnn_exec = cnn.bind(ctx=ctx, args=arg_arrays, args_grad=args_grad, grad_req='add')

    out_dict = dict(zip(cnn.list_outputs(), cnn_exec.outputs))
    data = cnn_exec.arg_dict['data']
    label = cnn_exec.arg_dict['softmax_label']
    return CNNModel(cnn_exec=cnn_exec, symbol=cnn, data=data, label=label, param_blocks=param_blocks)


def fit(m, text_iter, test_iter, batch_size,
        optimizer='rmsprop', max_grad_norm=5.0, learning_rate=0.0005, epoch=200, root='model/'):
    opt = mx.optimizer.create(optimizer)
    opt.lr = learning_rate
    updater = mx.optimizer.get_updater(opt)

    location_log = root + 'model.log'
    print location_log
    f_log = open(location_log, mode='w')
    for iteration in range(epoch):
        tic = time.time()
        num_correct = 0
        num_total = 0
        batch_num = text_iter.cnt / batch_size + 1
        time_for_data = 0
        real_train_time = 0.  # 包括取数据的时间， 但是不包括算accuracy的时间
        for _ in range(batch_num):
            train_tic = time.time()
            try:
                tic_ = time.time()
                batchX, batchY = text_iter.next_batch(batch_size)
                time_for_data += time.time() - tic_
            except Exception, e:
                logging.error("loading data error")
                logging.error(repr(e))
                continue
            # print 'x',batchX.shape,batchY.shape

            if batchX.shape[0] != batch_size:
                continue

            m.data[:] = batchX
            m.label[:] = batchY
            m.cnn_exec.forward(is_train=True)
            m.cnn_exec.backward()
            norm = 0
            for idx, weight, grad, name in m.param_blocks:
                grad /= batch_size
                l2_norm = mx.nd.norm(grad).asscalar()
                norm += l2_norm * l2_norm

            norm = math.sqrt(norm)
            for idx, weight, grad, name in m.param_blocks:
                if norm > max_grad_norm:
                    grad *= max_grad_norm / norm
                updater(idx, grad, weight)
                grad[:] = 0.0
            real_train_time += time.time() - train_tic
            num_correct += sum(batchY == np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
            num_total += len(batchY)

        if iteration % 50 == 0 and iteration > 0:
            opt.lr *= 0.5
            logging.info('reset learning rate to %g' % opt.lr)

        toc = time.time()
        train_time = toc - tic
        train_acc = num_correct * 100 / float(num_total)
        if (iteration + 1) % 1 == 0:
            prefix = root + 'cnn'
            m.symbol.save('%s-symbol.json' % prefix)
            save_dict = {'arg:%s' % k: v for k, v in m.cnn_exec.arg_dict.items() if k != 'embedding_weight'}
            save_dict.update({'aux:%s' % k: v for k, v in m.cnn_exec.aux_dict.items() if k != 'embedding_weight'})
            param_name = '%s-%04d.params' % (prefix, iteration)
            mx.nd.save(param_name, save_dict)
            logging.info('Saved checkpoint to %s' % param_name)

        if (iteration + 1) == epoch:
            save_dict_cpu = {k: v.copyto(mx.cpu()) for k, v in save_dict.items() if k != 'embedding_weight'}
            mx.nd.save(param_name + '.cpu', save_dict_cpu)
        num_correct = 0
        num_total = 0
        ps = []
        batch_num = test_iter.cnt / batch_size + 1
        y_dev_batch = []
        for _ in range(batch_num):
            try:
                batchX, batchY = test_iter.next_batch(batch_size)
            except Exception, e:
                logging.error(repr(e))
                continue
            if batchX.shape[0] != batch_size:
                continue
            y_dev_batch.extend(batchY)
            m.data[:] = batchX
            m.cnn_exec.forward(is_train=False)
            ps.extend(np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
            num_correct += sum(batchY == np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
            num_total += len(batchY)

        cat_res = evaluate(y_dev_batch, ps)
        dev_acc = num_correct * 100 / float(num_total)

        logging.info(
            'Iter [%d] Train: Time: %.3fs, Real Train Time: %.3f, Data Time: %.3fs, Training Accuracy: %.3f' % (
            iteration, train_time, real_train_time, time_for_data, train_acc))
        logging.info('--- Dev Accuracy thus far: %.3f' % dev_acc)

        line = 'Iter [%d] Train: Time: %.3fs, Training Accuracy: %.3f' % (iteration, train_time, train_acc)

        f_log.write(line.encode('utf-8') + '\n')
        f_log.flush()

        line = '--- Dev Accuracy thus far: %.3f' % dev_acc
        f_log.write(line.encode('utf-8') + '\n')
        f_log.flush()

        for cat, res in cat_res.items():
            logging.info('--- Dev Category %s P=%s,R=%s,F=%s' % (cat, res[0], res[1], res[2]))

            line = '--- Dev Category %s P=%s,R=%s,F=%s' % (cat, res[0], res[1], res[2])
            f_log.write(line.encode('utf-8') + '\n')
            f_log.flush()
    f_log.close()