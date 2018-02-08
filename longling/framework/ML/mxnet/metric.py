# coding: utf-8
# created by tongshiwei on 18-1-24

import numpy
from mxnet.metric import EvalMetric, check_label_shapes

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import log_loss


class LabelBuffMetric(EvalMetric):
    def __init__(self, name, output_names=None, label_names=None, pred_buff=[], true_buff=[], argmax=True):
        self.y_pred = pred_buff
        self.y_true = true_buff
        self.hash_code = None
        self.answer = None
        self.argmax = argmax
        super(LabelBuffMetric, self).__init__(name, output_names=output_names, label_names=label_names)

    def refresh_answer(self):
        if self.hash_code == self.hash_buff():
            return False
        else:
            return True

    def hash_buff(self):
        extra_hash = hash(str(self.y_pred[0])) + hash(str(self.y_pred[-1])) + hash(
            str(self.y_true[0])) + hash(str(self.y_true[-1])) if self.y_pred and self.y_true else 0
        return hash(len(self.y_pred) + len(self.y_true) + extra_hash)

    def reset(self):
        self.y_pred[:] = []
        self.y_true[:] = []
        self.hash_code = self.hash_buff()
        self.answer = None

    def _get(self):
        assert len(self.y_pred) == len(self.y_true)
        if len(self.y_pred) == 0:
            return self.name, float('nan')
        else:
            value = self.feval()
            if isinstance(value, list):
                names = []
                for i, v in enumerate(value):
                    names.append(self.name + '_%s' % i)
                return names, value
            else:
                return self.name, value

    def get(self):
        if self.refresh_answer():
            self.answer = self._get()
            self.hash_code = self.hash_buff()
            return self.answer
        else:
            return self.answer

    def feval(self):
        raise NotImplementedError

    def update(self, labels, preds):
        check_label_shapes(labels, preds)
        for label, pred in zip(labels, preds):
            pred = pred.asnumpy()
            if self.argmax:
                try:
                    y_pred = numpy.argmax(pred, axis=1)
                except Exception:
                    y_pred = numpy.argmax(pred)
            else:
                y_pred = pred
            y_true = label.asnumpy().astype('int32')
            self.y_pred += y_pred.tolist()
            self.y_true += y_true.tolist()


class Precision(LabelBuffMetric):
    def __init__(self, name="precision", output_names=None, label_names=None, pred_buff=[], true_buff=[], argmax=True):
        super(Precision, self).__init__(
            name,
            output_names=output_names,
            label_names=label_names,
            pred_buff=pred_buff,
            true_buff=true_buff,
            argmax=argmax,
        )

    def feval(self):
        return precision_score(self.y_true, self.y_pred)


class Recall(LabelBuffMetric):
    def __init__(self, name="recall", output_names=None, label_names=None, pred_buff=[], true_buff=[], argmax=True):
        super(Recall, self).__init__(
            name,
            output_names=output_names,
            label_names=label_names,
            pred_buff=pred_buff,
            true_buff=true_buff,
            argmax=argmax,
        )

    def feval(self):
        return recall_score(self.y_true, self.y_pred)


class F1(LabelBuffMetric):
    def __init__(self, name="recall", output_names=None, label_names=None, pred_buff=[], true_buff=[], argmax=True):
        super(F1, self).__init__(
            name,
            output_names=output_names,
            label_names=label_names,
            pred_buff=pred_buff,
            true_buff=true_buff,
            argmax=argmax,
        )

    def feval(self):
        return f1_score(self.y_true, self.y_pred)


class Accuracy(LabelBuffMetric):
    def __init__(self, name="accuracy", output_names=None, label_names=None, pred_buff=[], true_buff=[], argmax=True):
        super(Accuracy, self).__init__(
            name,
            output_names=output_names,
            label_names=label_names,
            pred_buff=pred_buff,
            true_buff=true_buff,
            argmax=argmax,
        )

    def feval(self):
        return accuracy_score(self.y_true, self.y_pred)


class CrossEntropy(LabelBuffMetric):
    def __init__(self, name="cross-entropy", output_names=None, label_names=None, pred_buff=[], true_buff=[], **kwargs):
        super(CrossEntropy, self).__init__(
            name,
            output_names=output_names,
            label_names=label_names,
            pred_buff=pred_buff,
            true_buff=true_buff,
            argmax=False,
        )

    def feval(self):
        return log_loss(self.y_true, self.y_pred)

    def update(self, labels, preds):
        check_label_shapes(labels, preds)
        for label, pred in zip(labels, preds):
            y_pred = pred.asnumpy().astype('float32')
            y_true = label.asnumpy().astype('int32')
            self.y_pred += y_pred.tolist()
            self.y_true += y_true.tolist()


class LabelMultiMetric(LabelBuffMetric):
    def __init__(self, name, output_names=None, label_names=None, pred_buff=[], true_buff=[], argmax=True):
        super(LabelMultiMetric, self).__init__(
            'multiMetric',
            output_names=output_names,
            label_names=label_names,
            pred_buff=pred_buff,
            true_buff=true_buff,
            argmax=argmax,
        )
        self.name = name if isinstance(name, list) else [name]

    def get_config(self):
        """Save configurations of metric. Can be recreated
        from configs with metric.create(**config)
        """
        config = self._kwargs.copy()
        config.update({
            'metric': self.__class__.__name__,
            'name': "_".join(self.name),
            'output_names': self.output_names,
            'label_names': self.label_names})

        return config

    def _get(self):
        raise NotImplementedError


class PRF(LabelMultiMetric):
    def __init__(self, name=['precision', 'recall', 'f1'], output_names=None, label_names=None, pred_buff=[],
                 true_buff=[], argmax=True):
        super(PRF, self).__init__(
            name,
            output_names=output_names,
            label_names=label_names,
            pred_buff=pred_buff,
            true_buff=true_buff,
            argmax=argmax,
        )

    def _get(self):
        precision, recall, f1, _ = precision_recall_fscore_support(self.y_true, self.y_pred)
        values = []
        names = []
        for name in self.name:
            for i, v in enumerate(eval(name)):
                names.append(name + "_%s" % i)
                values.append(v)
        return names, values
