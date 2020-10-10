# coding: utf-8
# 2020/4/15 @ tongshiwei

import numpy as np
from longling.ML.metrics import classification_report, regression_report


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
    assert result["macro_auc"] == 0.75

    y_true = np.array([0, 0, 1, 1])
    y_pred = [0, 0, 0, 1]
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    result = classification_report(y_true, y_pred, y_score=y_score)

    assert result["macro_avg"]["recall"] == 0.75
    assert result["macro_auc"] == 0.75
    assert "%.2f" % result["macro_aupoc"] == "0.83"

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


def test_regression():
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    result = regression_report(y_true, y_pred)
    assert result["mae"] == 0.5
    assert result["mse"] == 0.375

    result = regression_report(y_true, y_pred, metrics=["rmse"])
    assert "%.3f" % result["rmse"] == "0.612"

    y_true = [[0.5, 1], [-1, 1], [7, -6]]
    y_pred = [[0, 2], [-1, 2], [8, -5]]

    result = regression_report(y_true, y_pred, average_options="variance_weighted")
    assert "%.3f" % result["variance_weighted"]["r2"] == "0.938"
    assert "%.3f" % result["variance_weighted"]["evar"] == "0.983"

    result = regression_report(y_true, y_pred, average_options=[[0.3, 0.7]])
    assert "%.3f" % result["weighted"]["r2"] == "0.925"
    assert "%.3f" % result["weighted"]["evar"] == "0.990"
