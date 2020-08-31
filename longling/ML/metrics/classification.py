# coding: utf-8
# 2020/4/13 @ tongshiwei

import logging
from longling.lib.candylib import as_list
from collections import OrderedDict
from sklearn.metrics import (classification_report as cr,
                             roc_auc_score, average_precision_score, accuracy_score
                             )

from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import unique_labels
import functools


def multiclass2multilabel(y):
    """
    Parameters
    ----------
    y: 1d label indicator array
    """
    from sklearn.preprocessing import OneHotEncoder
    ret = []
    for _y in y:
        ret.append([_y])
    return OneHotEncoder().fit_transform(ret).todense()


def classification_report(y_true, y_pred=None, y_score=None, labels=None, metrics=None, sample_weight=None,
                          average_options=None, multiclass_to_multilabel=False, logger=logging, **kwargs):
    """
    Currently support binary and multiclasss classification.

    Parameters
    ----------
    y_true : list, 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : list or None, 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    y_score : array or None, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers). For binary
        y_true, y_score is supposed to be the score of the class with greater
        label.

    labels : array, shape = [n_labels]
        Optional list of label indices to include in the report.

    metrics: list of str,
        Support: precision, recall, f1, support, accuracy, auc, aupoc.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    average_options: str or list
        default to macro, choices (one or many): "micro", "macro", "samples", "weighted"

    multiclass_to_multilabel: bool

    logger

    Returns
    -------

    """
    if y_pred is not None:
        check_consistent_length(y_true, y_pred)
    if y_score is not None:
        check_consistent_length(y_true, y_score)

    assert y_pred is not None or y_score is not None

    average_options = set(as_list(average_options) if average_options else ["macro"])
    average_label_fmt = "{average}_avg"
    average_metric_fmt = "{average}_{metric}"

    if y_pred is not None:
        _unique_labels = unique_labels(y_true, y_pred)
    else:
        _unique_labels = unique_labels(y_true)

    labels = _unique_labels if labels is None else labels
    labels_set = set(labels)

    if not metrics:
        if y_pred is not None:
            metrics = [
                "accuracy", "precision", "recall", "f1",
            ]
        else:
            metrics = []
        if y_score is not None:
            metrics += [
                "auc", "aupoc",
            ]
        if y_pred is not None:
            metrics += ["support"]
    _metrics = set(metrics)

    ret = OrderedDict()

    if _metrics & {"precision", "recall", "f1", "support", "accuracy"}:
        logger.info("evaluate %s" % ",".join(_metrics & {"precision", "recall", "f1", "support", "accuracy"}))
        cr_result = cr(y_true, y_pred, labels=labels, sample_weight=sample_weight, output_dict=True)

        if "accuracy" in cr_result:
            acc = cr_result.pop("accuracy")
        else:
            acc = accuracy_score(y_true, y_pred)

        if "accuracy" in _metrics:
            ret["accuracy"] = acc

        for key, value in cr_result.items():
            ret[key] = {}

            for k in _metrics & {"precision", "recall", "f1", "support", "accuracy"}:
                _k = k if k != "f1" else "f1-score"
                if _k in value:
                    ret[key][k] = value[_k]

        for average in ["micro", "macro", "samples", "weighted"]:
            _label = average_label_fmt.format(average=average)
            __label = " ".join(_label.split("_"))
            _prefix = __label.split(" ")[0]
            if _prefix in average_options:
                ret[_label] = ret.pop(__label)
            elif __label in ret:
                ret.pop(__label)

    if "auc" in _metrics:
        logger.info("evaluate auc")

        assert y_score is not None, "when evaluate auc, y_score is required"

        func = functools.partial(roc_auc_score, y_score=y_score, sample_weight=sample_weight,
                                 **kwargs.get("auc", {"multi_class": 'ovo'}))

        if multiclass_to_multilabel:
            _y_true = multiclass2multilabel(y_true)
            auc_score = func(y_true=_y_true, average=None)
            for _label, score in enumerate(auc_score):
                if _label not in labels_set:
                    continue
                if str(_label) not in ret:
                    ret[str(_label)] = {}
                ret[str(_label)]["auc"] = score

            for average in average_options:
                auc_score = func(y_true=_y_true, average=average)
                _label = average_label_fmt.format(average=average)
                if _label not in ret:
                    ret[_label] = {}

                ret[_label]["auc"] = auc_score

        for average in average_options:
            auc_score = func(y_true=y_true, average=average)
            _label = average_metric_fmt.format(average=average, metric="auc")
            ret[_label] = auc_score

    if "aupoc" in _metrics:
        logger.info("evaluate aupoc")

        func = functools.partial(average_precision_score, y_score=y_score, sample_weight=sample_weight)

        if multiclass_to_multilabel:
            _y_true = multiclass2multilabel(y_true)
            aupoc = func(y_true=_y_true, average=None)
            for _label, score in enumerate(aupoc):
                if _label not in labels_set:
                    continue
                if str(_label) not in ret:
                    ret[str(_label)] = {}
                ret[str(_label)]["aupoc"] = score
            for average in average_options:
                _label = average_label_fmt.format(average=average)
                aupoc = func(y_true=_y_true, average=average)
                if _label not in ret:
                    ret[_label] = {}
                ret[_label]["aupoc"] = aupoc
        if len(_unique_labels) == 2:
            for average in average_options:
                aupoc = func(y_true=y_true)
                _label = average_metric_fmt.format(average=average, metric="aupoc")
                ret[_label] = aupoc

    logger.info("sorting metrics")
    _ret = OrderedDict()
    for key in ret:
        if isinstance(ret[key], dict):
            _ret[key] = OrderedDict()
            for k in metrics:
                if k in ret[key]:
                    _ret[key][k] = ret[key][k]
        else:
            _ret[key] = ret[key]

    return _ret
