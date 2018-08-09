# coding: utf-8
# created by tongshiwei on 18-1-27

import mxnet as mx
from mxnet.gluon import utils as gutil
import numpy as np

from longling.lib.candylib import as_list
from longling.framework.ML.MXnet.io_lib import SimpleBucketIter


def real_ctx(ctx, data_len):
    ctx = as_list(ctx)
    if data_len < len(ctx):
        ctx = ctx[:1]
    return ctx


def split_and_load(ctx, *args, **kwargs):
    ctx = real_ctx(ctx, len(args[0]))
    return zip(*[gutil.split_and_load(arg, ctx, **kwargs) for arg in args])


def form_shape(data_iter):
    return {d[0]: d.shape for d in data_iter.provide_data}


def get_fine_tune_model(symbol, label, arg_params, num_classes, layer_name='flatten0'):
    """
    symbol: the pretrained network symbol
    arg_params: the argument parameters of the pretrained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name + '_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, label=label, name='softmax')
    new_args = dict({k: arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)


class BasePredictor(object):
    def __init__(self, prefix, epoch=0, batch_size=128, load_optimizer_states=False, **kwargs):
        self.data_shapes = kwargs["data_shapes"]
        del kwargs["data_shapes"]
        self.data_names = kwargs.get("data_names", ["data", ])
        self.label_names = kwargs.get("label_names", [])
        kwargs.update({"data_names": self.data_names, "label_names": self.label_names})
        self.mod = mx.module.Module.load(prefix, epoch, load_optimizer_states, **kwargs)
        self.mod.bind(data_shapes=[('data', (batch_size,) + self.data_shapes)], force_rebind=True)
        self.batch_size = batch_size

    def predict(self, datas, label_tag=False):
        batch_size = min(len(datas), self.batch_size)
        datas = mx.io.NDArrayIter(data=np.asarray(datas), batch_size=batch_size, data_name="data")
        self.mod.bind(data_shapes=[('data', (batch_size,) + self.data_shapes)], force_rebind=True)
        preds = self.mod.predict(datas)
        if label_tag:
            return mx.ndarray.argmax(preds, axis=1)
        return preds

    def predict_prob(self, datas):
        probs = self.predict(datas)
        return probs.asnumpy()

    def predict_label(self, datas):
        labels = self.predict(datas, label_tag=True)
        return labels.asnumpy()


class RNNPredictor(object):
    def __init__(self, prefix, cells, sym, epoch=0, batch_size=128, buckets=[10, 20, 30], **kwargs):
        if buckets:
            _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(cells, prefix, epoch)
            self.mod = mx.module.BucketingModule(
                sym_gen=sym,
                default_bucket_key=max(buckets),
                **kwargs
            )
            self.mod.bind(
                data_shapes=[('data', (batch_size, max(buckets)))],
                label_shapes=[('label', (batch_size,))],
                force_rebind=True,
                for_training=False,
            )
            self.mod.set_params(arg_params=arg_params, aux_params=aux_params)
        else:
            raise NotImplementedError
        self.batch_size = batch_size
        self.buckets = buckets
        self.max_len = max(buckets)
        self.data_pad = kwargs.get("pad_num", 0) * self.max_len

    def predict(self, datas, label_tag=False):
        result = []
        datas, _, origin_idx = SimpleBucketIter.bucket_sort(datas, sorted_key=lambda x: len(x))
        data_buff = [[] for _ in self.buckets]
        assert SimpleBucketIter.bucket_distribution(
            data=datas,
            data_buffs=[data_buff],
            buckets=self.buckets,
            cut_off=True
        ) == 0
        for i, data in enumerate(data_buff):
            if not data:
                continue
            pad_num = SimpleBucketIter.padding(data, self.batch_size, [self.data_pad] * self.buckets[i])
            datas_iter = SimpleBucketIter(self.batch_size, data, buckets=self.buckets, for_predicting=True)
            res = self.mod.predict(datas_iter)
            if pad_num:
                res = res[:-pad_num]
            if label_tag:
                res = mx.ndarray.argmax(res, axis=1)
            result.extend(res.asnumpy())

        assert len(result) == len(origin_idx)
        result = mx.nd.array(result)
        map_idx = [0] * len(origin_idx)
        for i, idx in enumerate(origin_idx):
            map_idx[idx] = i

        preds = result[list(map_idx)]
        return preds

    def predict_prob(self, datas):
        probs = self.predict(datas)
        return probs.asnumpy()

    def predict_label(self, datas):
        labels = self.predict(datas, label_tag=True)
        return labels.asnumpy()
