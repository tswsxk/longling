# coding: utf-8
# 2020/4/15 @ tongshiwei

import pytest
from longling.ml.metrics import ranking_report


def test_ranking():
    y_true = [[1, 0], [0, 0, 1]]
    y_pred = [[0.75, 0.5], [1, 0.2, 0.1]]
    with pytest.raises(ValueError, match="Not enough value: 2 vs target 3"):
        ranking_report(y_true, y_pred, coerce="raise")

    with pytest.raises(ValueError, match="Not enough value: 2 vs target 3"):
        ranking_report(y_true, y_pred, coerce="raise", bottom=True)

    with pytest.raises(AssertionError, match="unknown coerce"):
        ranking_report(y_true, y_pred, coerce="err")
