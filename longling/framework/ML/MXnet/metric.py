# coding: utf-8
# created by tongshiwei on 18-1-24
# todo 优化注释结构

import numpy

import mxnet as mx
from mxnet.metric import EvalMetric, check_label_shapes

from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import log_loss


def _asnumpy(array):
    if isinstance(array, mx.nd.NDArray):
        return array.asnumpy()
    elif isinstance(array, (list, tuple)):
        return numpy.array(array)
    else:
        return array


class NoBufferMetric(EvalMetric):
    def __init__(self, name='NoLabelMetric', output_names=None,
                 label_names=None, **kwargs):
        super(NoBufferMetric, self).__init__(
            name=name,
            output_names=output_names,
            label_names=label_names,
            **kwargs
        )

    def feval(self, preds):
        raise NotImplementedError()

    def update(self, labels, preds):
        for pred in preds:
            pred = _asnumpy(pred)
            self.num_inst += len(pred)
            self.sum_metric += float(self.feval(pred))


class PairwiseMetric(NoBufferMetric):
    def __init__(self, name='pairwise', output_names=None,
                 label_names=None, **kwargs):
        super(NoBufferMetric, self).__init__(
            name=name,
            output_names=output_names,
            label_names=label_names,
            **kwargs
        )

    def feval(self, preds):
        return self.sum_metric + sum(preds)


class BufferMetric(EvalMetric):
    """
    带缓冲区的评测器
    由于mxnet自带的评价器会对测试集分批次评估，然后使用update进行简单的加权求和，所以部分指标是不准确的
    例如：precision，recall和f1，precision([1..i]) + precision([i+1,..n]) / 2 != precision([1..n])
    此抽象类定义了两个缓冲区来存储
    和mxnet里的函数不同，计算过程会延迟到需要输出结果的时候，因此要避免相同值的重新计算，因此引入了哈希值
    """

    def __init__(self, name, output_names=None, label_names=None, pred_buff=None, true_buff=None, argmax=True):
        """
        self.argmax 表示是否对pred要取max，适合pred数据为概率值或分数的情况

        Parameters
        ----------
        name
        output_names
        label_names
        pred_buff
        true_buff
        argmax
        """
        self.y_pred = pred_buff if pred_buff is not None else []
        self.y_true = true_buff if true_buff is not None else []
        self.hash_code = None
        self.answer = None
        self.argmax = argmax
        super(BufferMetric, self).__init__(name, output_names=output_names, label_names=label_names)

    @property
    def refresh_tag(self):
        """
        根据前后哈希值决定是否对结果进行更新
        """
        if self.hash_code == self.hash_buff:
            return False
        else:
            return True

    @property
    def hash_buff(self):
        """
        缓冲区哈希值
        """

        extra_hash = hash(str(self.y_pred[0])) + hash(str(self.y_pred[-1])) + hash(
            str(self.y_true[0])) + hash(str(self.y_true[-1])) if self.y_pred and self.y_true else 0
        return hash(len(self.y_pred) + len(self.y_true) + extra_hash)

    def reset(self):
        '''
        清空缓冲区
        :return:
        '''
        self.y_pred[:] = []
        self.y_true[:] = []
        self.hash_code = self.hash_buff
        self.answer = None

    def _get(self):
        '''
        包装结果,返回符合mxnet.Metric格式的结果
        :return:
        '''
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
        '''
        mxnet.EvalMetric API
        API
        获取结果，根据哈希值决定是否重算
        :return: self.answer
        '''
        if self.refresh_tag:
            self.answer = self._get()
            self.hash_code = self.hash_buff
            return self.answer
        else:
            return self.answer

    def feval(self):
        '''
        结果计算
        :return:
        '''
        raise NotImplementedError

    def update(self, labels, preds):
        '''
        mxnet.EvalMetric API

        更新缓冲区
        增加新的labels和preds

        :param labels:
        :param preds:
        :return:
        '''
        check_label_shapes(labels, preds)
        for label, pred in zip(labels, preds):
            pred = _asnumpy(pred)
            if self.argmax:
                try:
                    y_pred = numpy.argmax(pred, axis=1)
                except Exception:
                    y_pred = numpy.argmax(pred)
            else:
                y_pred = pred
            y_true = _asnumpy(label).astype('int32')
            self.y_pred += y_pred.tolist()
            self.y_true += y_true.tolist()


class Precision(BufferMetric):
    def __init__(self, name="Precision", output_names=None, label_names=None, pred_buff=None, true_buff=None,
                 argmax=True):
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


class Recall(BufferMetric):
    def __init__(self, name="Recall", output_names=None, label_names=None, pred_buff=None, true_buff=None, argmax=True):
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


class F1(BufferMetric):
    def __init__(self, name="F1", output_names=None, label_names=None, pred_buff=None, true_buff=None, argmax=True):
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


class Accuracy(BufferMetric):
    def __init__(self, name="Accuracy", output_names=None, label_names=None, pred_buff=None, true_buff=None,
                 argmax=True):
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


class AUC(BufferMetric):
    def __init__(self, name="auc", output_names=None, label_names=None, pred_buff=None, true_buff=None,
                 argmax=True):
        super(AUC, self).__init__(
            name,
            output_names=output_names,
            label_names=label_names,
            pred_buff=pred_buff,
            true_buff=true_buff,
            argmax=argmax,
        )

    def feval(self):
        return roc_auc_score(self.y_true, self.y_pred)


class R2Square(BufferMetric):
    def __init__(self, name="R2Square", output_names=None, label_names=None, pred_buff=None, true_buff=None,
                 argmax=True):
        super(R2Square, self).__init__(
            name,
            output_names=output_names,
            label_names=label_names,
            pred_buff=pred_buff,
            true_buff=true_buff,
            argmax=argmax,
        )

    def feval(self):
        return r2_score(self.y_true, self.y_pred)


class CrossEntropy(BufferMetric):
    def __init__(self, name="CrossEntropy", output_names=None, label_names=None, pred_buff=None, true_buff=None):
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
            y_pred = _asnumpy(pred).astype('float32')
            y_true = _asnumpy(label).astype('int32')
            self.y_pred += y_pred.tolist()
            self.y_true += y_true.tolist()


class LabelMultiMetric(BufferMetric):
    def __init__(self, name, output_names=None, label_names=None, pred_buff=None, true_buff=None, argmax=True):
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
        """
        Save configurations of metric.Can be recreated from configs with metric.create(**config)
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

    def feval(self):
        pass


class PRF(LabelMultiMetric):
    def __init__(self, name=None, output_names=None, label_names=None, pred_buff=None,
                 true_buff=None, argmax=True):
        name = ['precision', 'recall', 'f1'] if name is None else name

        super(PRF, self).__init__(
            name,
            output_names=output_names,
            label_names=label_names,
            pred_buff=pred_buff,
            true_buff=true_buff,
            argmax=argmax,
        )

    def _get(self):
        precision, recall, f1, _ = self.feval()
        values = []
        names = []
        for name in self.name:
            for i, v in enumerate(eval(name)):
                names.append(name + "_%s" % i)
                values.append(v)
        return names, values

    def feval(self):
        return precision_recall_fscore_support(self.y_true, self.y_pred)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
