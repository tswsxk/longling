# coding: utf-8
# 2020/4/17 @ tongshiwei

import numpy as np

from longling.ML.metrics import classification_report, regression_report
from longling.ML.toolkit.formatter import eval_format, EpochEvalFMT, EpisodeEvalFMT

a = """
test_eval_format	Iteration [10]	Train Time-137.123s	Loss - loss: 2.59	{'a': 123}
           precision    recall        f1  support
0           0.666667  0.666667  0.666667        3
1           0.500000  0.500000  0.500000        2
macro_avg   0.583333  0.583333  0.583333        5
accuracy: 0.600000
"""
b = """
evar: 0.957173	mse: 0.375000	rmse: 0.612372	mae: 0.500000	r2: 0.948608
"""


def test_eval_format():
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0])
    result = classification_report(y_true, y_pred)

    assert eval_format(
        tips="test_eval_format",
        iteration=10,
        train_time=137.1234,
        loss_name_value={"loss": 2.59},
        eval_name_value=result,
        extra_info={"a": 123}
    ) == a.strip("\n")

    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    result = regression_report(y_true, y_pred)

    assert eval_format(
        eval_name_value=result,
        dump_file=None,
        keep="msg"
    ) == b.strip()

    assert eval_format(
        eval_name_value=result,
        dump_file=None,
        keep=None
    ) == b.strip()

    assert eval_format(
        eval_name_value=result,
        dump_file=None,
        keep={"msg", "data"}
    )[0] == b.strip()


def test_eval_fmt_variants():
    eval_fmt = EpochEvalFMT.format(
        iteration=30, keep="data"
    )
    assert eval_fmt["Epoch"] == 30

    eval_fmt = EpisodeEvalFMT.format(
        iteration=10, keep="data"
    )
    assert eval_fmt["Episode"] == 10
