# coding: utf-8
# 2020/4/15 @ tongshiwei

import pytest
import numpy as np
from longling.ML.metrics import classification_report
from longling.ML.toolkit.formatter import EvalFMT


def test_classification():
    # for binary
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0])
    result = classification_report(y_true, y_pred)
    assert result["accuracy"] == 0.6
    assert result["1"]["f1"] == 0.5

    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    result = classification_report(y_true, y_score=y_score, metrics=["auc", "poauc"])
    assert result["macro auc"] == 0.75

    y_true = np.array([0, 0, 1, 1])
    y_pred = [0, 0, 0, 1]
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    result = classification_report(y_true, y_pred, y_score=y_score)

    assert result["macro avg"]["recall"] == 0.75
    assert result["macro auc"] == 0.75
    assert "%.2f" % result["macro aupoc"] == "0.83"

    # for multiclass
    y_true = [0, 1, 2, 2, 2]
    y_pred = [0, 0, 2, 2, 1]
    result = classification_report(y_true, y_pred)
    assert result["accuracy"] == 0.6
    assert result["0"]["precision"] == 0.5

    # multiclass in multilabel
    y_true = np.array([0, 0, 1, 1, 2, 1])
    y_pred = np.array([2, 1, 0, 2, 1, 0])
    y_score = np.array([
        [0.15, 0.4, 0.45],
        [0.1, 0.9, 0.0],
        [0.33333, 0.333333, 0.333333],
        [0.15, 0.4, 0.45],
        [0.1, 0.9, 0.0],
        [0.33333, 0.333333, 0.333333]
    ])

    classification_report(
        y_true, y_pred, y_score,
        multiclass_to_multilabel=True,
        metrics=["aupoc"]
    )

    classification_report(
        y_true, y_pred, y_score,
        multiclass_to_multilabel=True,
        metrics=["auc", "aupoc"]
    )

    y_true = np.array([0, 1, 1, 1, 2, 1])
    y_pred = np.array([2, 1, 0, 2, 1, 0])
    y_score = np.array([
        [0.45, 0.4, 0.15],
        [0.1, 0.9, 0.0],
        [0.33333, 0.333333, 0.333333],
        [0.15, 0.4, 0.45],
        [0.1, 0.9, 0.0],
        [0.33333, 0.333333, 0.333333]
    ])

    classification_report(
        y_true, y_pred,
        y_score,
        multiclass_to_multilabel=True,
    )

    classification_report(
        y_true, y_pred,
        y_score,
        labels=[0, 1],
        multiclass_to_multilabel=True,
    )


if __name__ == '__main__':
    y_true = np.array([0, 1, 1, 1, 2, 1])
    y_pred = np.array([2, 1, 0, 2, 1, 0])
    y_score = np.array([
        [0.45, 0.4, 0.15],
        [0.1, 0.9, 0.0],
        [0.33333, 0.333333, 0.333333],
        [0.15, 0.4, 0.45],
        [0.1, 0.9, 0.0],
        [0.33333, 0.333333, 0.333333]
    ])

    result = classification_report(
        y_true, y_pred,
        y_score,
        labels=[0, 1],
        multiclass_to_multilabel=True,
    )

    print(EvalFMT.format(eval_name_value=result)[0])
