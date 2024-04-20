# coding: utf-8
# 2020/4/14 @ tongshiwei

import pytest
from longling.utils.formatter import series_format, table_format


def test_series_format():
    series_format({"a": 12.3, "b": 3, "c": 4, "d": 5}, digits=1)
    series_format({"a": 12.3, "b": 3, "c": 4, "d": 5}, digits=1, col=3)

    with pytest.raises(TypeError):
        table_format("hello world")
