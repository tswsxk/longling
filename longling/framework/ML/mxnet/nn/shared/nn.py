# coding: utf-8
# create by tongshiwei on 2017/10/23

from __future__ import division

import json
import logging
import math
import os
import pickle
import time
from abc import ABCMeta, abstractmethod
from collections import namedtuple

import mxnet as mx
import numpy as np
from longling.lib.utilog import config_logging
from tqdm import tqdm

from longling.framework.ML.universe.metrics.model_eval import evaluate
from longling.lib.stream import wf_open, wf_close
from longling.base import *

NNModel = namedtuple('Model', ['model_exec', 'symbol', 'data', 'label', 'param_blocks', 'args_grad'])


class NN():
    __metaclass__ = ABCMeta

    def __init__(self, logger=None):
        self.logger = self.get_logger(logger)

    def get_logger(self, logger=None):
        if logger is None or isinstance(logger, str):
            logger = config_logging(
                logger=self.__class__.__name__,
                level=logging.INFO,
                console_log_level=logging.INFO,
                propagate=False,
            )
        else:
            logger = logger

        return logger

    @abstractmethod
    def get_symbol_without_loss(self, **kwargs):
        pass

    @abstractmethod
    def get_symbol(self, **kwargs):
        pass

    @abstractmethod
    def get_model(self, **kwargs):
        pass

    @abstractmethod
    def record_parameters(self, **extend_prameters):
        pass

    def fit(self, model, train_iter, test_iter, batch_size,
            optimizer='rmsprop', max_grad_norm=5.0, learning_rate=0.0005, lr_update_rate=50, start_epoch=0,
            epoch=200, model_dir='model/', saved_epoch=1):
        fit(
            model=model,
            train_iter=train_iter,
            test_iter=test_iter,
            batch_size=batch_size,
            optimizer=optimizer,
            max_grad_norm=max_grad_norm,
            learning_rate=learning_rate,
            lr_update_rate=lr_update_rate,
            epoch=epoch,
            model_dir=model_dir,
            saved_epoch=saved_epoch,
            logger=self.logger,
        )

    @abstractmethod
    def set_predictor(self, batch_size, ctx, checkpoint):
        pass

    @abstractmethod
    def predictProba(self, datas):
        return []

    def clean_predictor(self):
        if hasattr(self, 'predictor'):
            del self.predictor

    def predict(self, datas):
        if not hasattr(self, 'predictor'):
            raise Exception('should set predictor before predict data - run (instance of nn).set_predictor')
        probs = self.predictProba(datas)
        labels = [r.argmax() for r in probs]
        return labels, probs

    def predict_label(self, datas):
        if not hasattr(self, 'predictor'):
            raise Exception('should set predictor before predict data - run (instance of nn).set_predictor')
        return [r.argmax() for r in self.predict_proba(datas)]

    def predict_proba(self, datas):
        if not hasattr(self, 'predictor'):
            raise Exception('should set predictor before predict data - run (instance of nn).set_predictor')
        return self.predictProba(datas)

    def save_model(self, location):
        NN.checkDir(location)
        logger = self.logger
        del self.logger
        with open(location, "w") as wf:
            wf.write(pickle.dumps(self))
        logger.info("save model to %s" % location)
        self.logger = logger

    @staticmethod
    def load_model(location, logger=None):
        with open(location) as f:
            data = f.read()
            obj = pickle.loads(data)
        obj.logger = obj.get_logger(logger)
        return obj

    @staticmethod
    def checkDir(path, mode=0o777):
        dirname = os.path.dirname(path)
        if os.path.exists(dirname):
            return
        os.makedirs(dirname, mode)

    @staticmethod
    def plot_network(nn_symbol, save_path="plot/network", shape=None, node_attrs={}, show_tag=False):
        plot_network(
            nn_symbol=nn_symbol,
            save_path=save_path,
            shape=shape,
            node_attrs=node_attrs,
            show_tag=show_tag,
        )

    def form_checkpoint(self, model_dir, checkpoint=None):
        return form_checkpoint(model_dir, checkpoint, self.logger)

    def form_ctx(self, ctx=-1):
        return form_ctx(ctx, self.logger)


def form_checkpoint(model_dir, checkpoint=None, logger=logging):
    if checkpoint is None:
        return checkpoint
    if not isinstance(checkpoint, int):
        try:
            checkpoint = int(checkpoint)
        except:
            logger.error('invalid checkpoint format!')
            raise Exception('invalid checkpoint format!')
    checkpoint = os.path.join(model_dir, '%s-%04d.params' % ('cnn', checkpoint))
    if not os.path.exists(checkpoint):
        logger.error('checkpoint does not exist!')
        raise Exception('checkpoint does not exist!')
    return checkpoint


def form_ctx(ctx=-1, logger=logging):
    if int(ctx) >= 0:
        logger.info("process_fit use gpu-%s" % ctx)
        ctx = mx.gpu(ctx)
    else:
        ctx = mx.cpu(0)
        logger.info("process_fit use cpu")
    return ctx


def add_loss_layer(layer, loss='softmax'):
    input_y = mx.sym.Variable('label')
    if loss == 'softmax':
        return mx.sym.SoftmaxOutput(data=layer, label=input_y, name='softmax')
    else:
        raise Exception('unknown loss function')


def get_model(ctx, nn_symbol, feature_num, batch_size, vec_size=-1, channel=1, return_grad=False,
              checkpoint=None, logger=logging):
    input_shapes = dict()
    if vec_size is -1:
        if channel is 1 or channel is -1:
            input_shapes['data'] = (batch_size, feature_num)

    else:
        input_shapes['data'] = (batch_size, channel, feature_num, vec_size)

    return _get_model(
        ctx=ctx,
        nn_symbol=nn_symbol,
        input_shapes=input_shapes,
        return_grad=return_grad,
        checkpoint=checkpoint,
        logger=logger,
    )


def _get_model(ctx, nn_symbol, input_shapes, return_grad=False, checkpoint=None, logger=logging):
    arg_names = nn_symbol.list_arguments()
    arg_shape, out_shape, aux_shape = nn_symbol.infer_shape(**input_shapes)

    # init space for grads
    args_grad = {}
    for name, shape in zip(arg_names, arg_shape):
        if name in ('label', 'data', 'embedding_weight'):
            continue
        args_grad[name] = mx.nd.zeros(shape, ctx)

    if return_grad:
        args_grad['data'] = mx.nd.zeros(input_shapes['data'], ctx)

    arg_dict = {}
    if checkpoint != None:
        logger.info("checkpoint: %s" % checkpoint)
        p = mx.nd.load(checkpoint)
        for k, v in p.items():
            if k.startswith('arg:'):
                k = k[4:]
                arg_dict[k] = mx.nd.array(v.asnumpy(), ctx)
    initializer = mx.initializer.Uniform(0.1)
    arg_arrays = []
    param_blocks = []

    for i, (name, shape) in enumerate(zip(arg_names, arg_shape)):
        if name in ('label', 'data'):
            arg_arrays.append(mx.nd.zeros(shape, ctx))
            continue
        if name not in arg_dict:
            arg_dict[name] = mx.nd.ones(shape, ctx)
            initializer(mx.init.InitDesc(name), arg_dict[name])
        arg_arrays.append(arg_dict[name])
        param_blocks.append((i, arg_dict[name], args_grad[name], name))
    nn_exec = nn_symbol.bind(ctx=ctx, args=arg_arrays, args_grad=args_grad, grad_req='add')

    # out_dict = dict(zip(cnn_symbol.list_outputs(), cnn_exec.outputs))
    data = nn_exec.arg_dict['data']
    label = nn_exec.arg_dict.get('label', None)
    return NNModel(model_exec=nn_exec, symbol=nn_symbol, data=data, label=label, param_blocks=param_blocks,
                   args_grad=args_grad)


def get_embeding_model(ctx, nn_symbol, embedding, feature_num, batch_size, return_grad=False,
                       checkpoint=None, logger=logging):
    arg_names = nn_symbol.list_arguments()
    input_shapes = dict()
    input_shapes['data'] = (batch_size, feature_num)
    arg_shape, out_shape, aux_shape = nn_symbol.infer_shape(**input_shapes)

    # init space for grads
    args_grad = {}
    for name, shape in zip(arg_names, arg_shape):
        if name in ('label', 'data', 'embedding_weight'):
            continue
        args_grad[name] = mx.nd.zeros(shape, ctx)

    if return_grad:
        args_grad['data'] = mx.nd.zeros(input_shapes['data'], ctx)

    arg_dict = {}
    if checkpoint != None:
        logger.info("checkpoint: %s" % checkpoint)
        p = mx.nd.load(checkpoint)
        for k, v in p.items():
            if k.startswith('arg:'):
                k = k[4:]
                arg_dict[k] = mx.nd.array(v.asnumpy(), ctx)
    initializer = mx.initializer.Uniform(0.1)
    arg_arrays = []
    param_blocks = []

    for i, (name, shape) in enumerate(zip(arg_names, arg_shape)):
        if name in ('label', 'data'):
            arg_arrays.append(mx.nd.zeros(shape, ctx))
            continue
        if name == "embedding_weight":
            arg_arrays.append(mx.nd.array(embedding, ctx))
            continue
        if name not in arg_dict:
            arg_dict[name] = mx.nd.zeros(shape, ctx)
            initializer(mx.init.InitDesc(name), arg_dict[name])
        arg_arrays.append(arg_dict[name])
        param_blocks.append((i, arg_dict[name], args_grad[name], name))
    nn_exec = nn_symbol.bind(ctx=ctx, args=arg_arrays, args_grad=args_grad, grad_req='add')

    # out_dict = dict(zip(cnn_symbol.list_outputs(), cnn_exec.outputs))
    data = nn_exec.arg_dict['data']
    label = nn_exec.arg_dict['label']
    return NNModel(model_exec=nn_exec, symbol=nn_symbol, data=data, label=label, param_blocks=param_blocks)


def fit(model, train_iter, test_iter, batch_size,
        optimizer='rmsprop', max_grad_norm=5.0, learning_rate=0.0005, lr_update_rate=50, start_epoch=0, epoch=200,
        model_dir='model/', saved_epoch=1, logger=None):
    # set logger
    if logger is None:
        logger = logging

    logger.info("start fitting")

    # set optimizer
    opt = mx.optimizer.create(optimizer)
    opt.lr = learning_rate
    updater = mx.optimizer.get_updater(opt)

    # set model_log
    model_log = model_dir + 'model.log'
    logger.info("model log location is %s" % model_log)
    f_log = wf_open(model_log)

    result_log = model_dir + 'result.log'
    logger.info("result log location is %s" % result_log)
    r_log = wf_open(result_log)

    # training
    for iteration in range(start_epoch, epoch):
        tic = time.time()
        num_correct = 0
        num_total = 0
        batch_num = (train_iter.cnt + batch_size - 1) // batch_size
        time_for_data = 0
        real_train_time = 0.  # 包括取数据的时间， 但是不包括算accuracy的时间

        train_iter.reset()
        test_iter.reset()

        # batch training
        for _ in tqdm(range(batch_num)):
            train_tic = time.time()

            # loading data
            try:
                tic_ = time.time()
                batchX, batchY = train_iter.next(batch_size)
                time_for_data += time.time() - tic_
            except Exception as e:
                logger.error("loading data error")
                logger.error(repr(e))
                continue

            if batchX.shape[0] != batch_size:
                continue

            model.data[:] = batchX
            model.label[:] = batchY
            model.model_exec.forward(is_train=True)
            model.model_exec.backward()
            norm = 0
            for idx, weight, grad, name in model.param_blocks:
                grad /= batch_size
                l2_norm = mx.nd.norm(grad).asscalar()
                norm += l2_norm * l2_norm

            norm = math.sqrt(norm)
            for idx, weight, grad, name in model.param_blocks:
                if norm > max_grad_norm:
                    grad *= max_grad_norm / norm
                updater(idx, grad, weight)
                grad[:] = 0.0
            real_train_time += time.time() - train_tic
            num_correct += sum(batchY == np.argmax(model.model_exec.outputs[0].asnumpy(), axis=1))
            num_total += len(batchY)

        # reset learning rate
        if iteration % lr_update_rate == 0 and iteration > 0:
            opt.lr *= 0.5
            logger.info('reset learning rate to %g' % opt.lr)

        toc = time.time()
        train_time = toc - tic
        train_acc = num_correct * 100 / float(num_total)

        # saving iteration checkpoint
        if (iteration + 1) % saved_epoch == 0:
            prefix = model_dir + 'cnn'
            model.symbol.save('%s-symbol.json' % prefix)

            save_dict = {'arg:%s' % k: v for k, v in model.model_exec.arg_dict.items() if k != 'embedding_weight'}
            save_dict.update({'aux:%s' % k: v for k, v in model.model_exec.aux_dict.items() if k != 'embedding_weight'})
            param_name = '%s-%04d.params' % (prefix, iteration)
            mx.nd.save(param_name, save_dict)
            logger.info('Saved checkpoint to %s' % param_name)

            # save_dict_cpu = {k: v.copyto(mx.cpu()) for k, v in save_dict.items() if k != 'embedding_weight'}
            # mx.nd.save(param_name + '.cpu', save_dict_cpu)
            # logger.info('Saved checkpoint to %s' % param_name + '.cpu')

        # saving final checkpoint to cpu backup
        if (iteration + 1) == epoch:
            prefix = model_dir + 'cnn'
            model.symbol.save('%s-symbol.json' % prefix)

            save_dict = {'arg:%s' % k: v for k, v in model.model_exec.arg_dict.items() if k != 'embedding_weight'}
            save_dict.update({'aux:%s' % k: v for k, v in model.model_exec.aux_dict.items() if k != 'embedding_weight'})
            param_name = '%s-%04d.params' % (prefix, iteration)
            mx.nd.save(param_name, save_dict)
            logger.info('Saved final checkpoint to %s' % param_name)

            # save_dict_cpu = {k: v.copyto(mx.cpu()) for k, v in save_dict.items() if k != 'embedding_weight'}
            # mx.nd.save(param_name + '.cpu', save_dict_cpu)
            # logger.info('Saved final checkpoint to %s' % param_name + '.cpu')

        # evaluate model in batch
        num_correct = 0
        num_total = 0
        ps = []
        batch_num = (test_iter.cnt + batch_size - 1) // batch_size
        y_dev_batch = []
        for _ in range(batch_num):
            try:
                batchX, batchY = test_iter.next(batch_size)
            except Exception as e:
                logging.error(repr(e))
                continue
            if batchX.shape[0] != batch_size:
                continue
            y_dev_batch.extend(batchY)
            model.data[:] = batchX
            model.model_exec.forward(is_train=False)
            ps.extend(np.argmax(model.model_exec.outputs[0].asnumpy(), axis=1))
            num_correct += sum(batchY == np.argmax(model.model_exec.outputs[0].asnumpy(), axis=1))
            num_total += len(batchY)

        # evaluate result
        cat_res = evaluate(gold_list=y_dev_batch, predict_list=ps)
        dev_acc = num_correct * 100 / float(num_total)

        # report result
        logger.info(
            'Iter [%d] Train: Time: %.3fs, Real Train Time: %.3fs, Data Time: %.3fs, Training Accuracy: %.3f' % (
                iteration, train_time, real_train_time, time_for_data, train_acc))
        logger.info('--- Dev Accuracy thus far: %.3f' % dev_acc)

        line = 'Iter [%d] Train: Time: %.3fs, Training Accuracy: %.3f' % (iteration, train_time, train_acc)

        f_log.write(tostr(line) + '\n')
        f_log.flush()

        line = '--- Dev Accuracy thus far: %.3f' % dev_acc
        f_log.write(tostr(line) + '\n')
        f_log.flush()

        result = {
            'iteration': iteration,
            'train_time': train_time,
            'real_train_time': real_train_time,
            'train_acc': train_acc,
            'acc': dev_acc,
            'prf': {}
        }
        for cat, res in cat_res.items():
            logging.info('--- Dev Category %s P=%s,R=%s,F=%s' % (cat, res[0], res[1], res[2]))

            line = '--- Dev Category %s P=%s,R=%s,F=%s' % (cat, res[0], res[1], res[2])

            f_log.write(tostr(line) + '\n')
            f_log.flush()

            result['prf'][tostr(cat)] = [res[0], res[1], res[2]]

        r_log.write(json.dumps(result, ensure_ascii=False) + '\n')
        r_log.flush()

    wf_close(f_log)
    wf_close(r_log)


def plot_network(nn_symbol, save_path="plot/network", shape=None, node_attrs={}, show_tag=False):
    graph = mx.viz.plot_network(nn_symbol, shape=shape, node_attrs=node_attrs)

    assert save_path
    graph.render(save_path)

    if show_tag:
        graph.view()
