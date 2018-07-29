# coding: utf-8

from __future__ import print_function

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, average_precision_score, recall_score, f1_score
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.metrics import classification_report


from sklearn.metrics import precision_recall_fscore_support


def prf_score(y_true, y_pred, sample_weight=None):
    """

    Parameters
    ----------
    y_true
    y_pred
    sample_weight

    Returns
    -------

    :param y_true:
    :param y_pred:
    :param sample_weight:
    :param accelerate: None or '', process, thread
    :return:
    """
    labels = set(y_true)
    prf = {label: [0] * 3 for label in labels}

    ps, rs, fs, _ = precision_recall_fscore_support(y_true, y_pred, sample_weight=sample_weight)
    for label, p, r, f in zip(labels, ps, rs, fs):
        prf[label] = [p, r, f]

    return prf

if __name__ == '__main__':

    y_true = [1, 1, 1, 0, 2, 2] * 100000
    y_pred = [1, 1, 0, 0, 0, 2] * 100000

    test_func = lambda x: print(x.__name__, x(y_true, y_pred))

    # multilabel - multilabel
    test_func(classification_report)
    test_func(accuracy_score)
    test_func(prf_score)
    print(precision_score(y_true, y_pred, average=None))
    print(recall_score(y_true, y_pred, average=None))
    print(f1_score(y_true, y_pred, average=None))

    # label - pro
    y_true = [1, 1, 0, 0] * 1000
    y_pred = [0.20, 0.23, 0.77, 0.1] * 1000
    test_func(roc_auc_score)
    # curve for plot
    test_func(roc_curve) # ?

    # ordered, little useful
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]
    test_func(auc)

    # label - label
    y_true = [0, 0, 1, 1] * 1000
    y_pred = [1, 0, 1, 1] * 1000
    test_func(log_loss)  # ?
    test_func(average_precision_score)





