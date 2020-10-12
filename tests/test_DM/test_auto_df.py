# coding: utf-8
# create by tongshiwei on 2020-10-12

import pandas as pd
import pytest

from longling.DM import auto_fti
from longling.DM.pdh import RegexDict


def test_regex_dict():
    rd = RegexDict({"^d1$": 1, "^d2$": 3})
    with pytest.raises(KeyError):
        print(rd["z3"])


def test_auto_df():
    import numpy as np

    a = pd.DataFrame([
        {"a": 1, "b": "2008-03-03", "c": 0.0, "d1": np.nan, "d2": "x", "e": "h", "f": 5},
        {"a": 1, "b": "2008-03-04", "c": 1.0, "d1": "0"},
        {"a": 3, "b": "2008-03-05", "c": 3.0, "d1": "1", "f": 3},
        {"a": 3, "b": "2008-03-06", "c": 3.0, "d1": "1"},
        {"a": np.nan, "b": "2008-03-07", "c": 4.0, "d1": "2", "d2": "y", "e": "p"},
    ])
    a.info()
    auto_fti(
        a,
        category_columns=["d1"],
        columns_to_code=["$regex:d", "e"],
        category_code_offset=RegexDict({"^d1$": 1, "^d2$": 3}, default_value=5),
        datetime_columns=["b"],
        na_fill_mode="mode",
        verbose=True
    )
    a.info()
    print(a)
