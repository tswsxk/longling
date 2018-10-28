# coding: utf-8
# created by tongshiwei on 18-2-5

import re
import codecs
import json
import logging

import mxnet as mx
import numpy as np

from tqdm import tqdm

from longling.base import string_types
from longling.lib.stream import wf_open


class Evaluator(object):
    def __init__(self, metrics=None, logger=logging, log_f=None, **kwargs):
        if not isinstance(metrics, mx.metric.EvalMetric):
            self.metrics = mx.metric.create(metrics) if metrics is not None else None
        self.logger = logger
        if log_f is not None and isinstance(log_f, string_types):
            # clean file
            wf_open(log_f).close()
        self.log_f = log_f

    def evaluate(self, data_iterator, net, stage=""):
        raise NotImplementedError

    @staticmethod
    def format_eval_res(epoch, eval_name_value, loss_name_value=None, train_time=None, output=True, *args, **kwargs):
        assert isinstance(eval_name_value, dict), "input should be a dict"
        data = {}
        msg = 'Epoch [%d]:' % epoch
        data['iteration'] = epoch
        if train_time is not None:
            msg += "\tTrain Time-%.3fs" % train_time
            data['train_time'] = train_time

        if loss_name_value is not None:
            msg += "\tLoss - "
            for n, v in loss_name_value:
                msg += "%s: %f" % (n, v)

        name_value = dict(eval_name_value)
        multi_class_pattern = re.compile(".+_\d+")

        prf = {}
        eval_ids = set()

        for name, value in sorted(name_value.items()):
            if multi_class_pattern.match(name) is not None:
                try:
                    eval_id, class_id = name.split("_")
                except ValueError:
                    continue
                if class_id not in prf:
                    prf[class_id] = {}
                prf[class_id][eval_id] = value
                eval_ids.add(eval_id)
            else:
                msg += "\tValidation %s: %f" % (name, value)
                data[name] = value

        if prf:
            avg = {eval_id: [] for eval_id in eval_ids}
            for class_id in [str(k) for k in sorted([int(k) for k in prf.keys()])]:
                for eval_id, values in avg.items():
                    values.append(prf[class_id][eval_id])
                msg += "\n"
                msg += "--- Category %s" % class_id
                msg_res = sorted(prf[class_id].items(), reverse=True)
                msg += ("\t{}={:.10f}" * len(prf[class_id])).format(*sum(msg_res, ()))
            avg = {eval_id: np.average(values) for eval_id, values in avg.items()}
            msg += "\n"
            msg += "--- Category_Avg "
            msg_res = sorted(avg.items(), reverse=True)
            msg += ("\t{}={:.10f}" * len(avg)).format(*sum(msg_res, ()))
            prf['avg'] = avg
            data['prf'] = prf

        if output:
            logger = kwargs.get('logger', logging)
            logger.info(msg)
            if kwargs.get('log_f', None) is not None:
                log_f = kwargs['log_f']
                try:
                    if log_f is not None and isinstance(log_f, string_types):
                        log_f = codecs.open(log_f, "a", encoding="utf-8")
                        print(json.dumps(data, ensure_ascii=False), file=log_f)
                        log_f.close()
                    else:
                        print(json.dumps(data, ensure_ascii=False), file=log_f)
                except Exception as e:
                    logger.warning(e)
        return msg, data


class ClassEvaluator(Evaluator):
    def evaluate(self, data_iterator, net, stage="", model_ctx=mx.cpu()):
        # deprecated
        for i, (data, label) in enumerate(tqdm(data_iterator, desc="%s evaluating" % stage)):
            data = data.as_in_context(model_ctx)
            label = label.as_in_context(model_ctx)
            output = net(data)
            predictions = mx.nd.argmax(output, axis=1)
            self.metrics.update(preds=predictions, labels=label)
        return self.metrics.get_name_value()
