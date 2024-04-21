# coding: utf-8
# create by tongshiwei on 2020-10-12

import numpy as np
import pandas as pd
import pytest

from longling.dm import auto_fti
from longling.dm.pdh import RegexDict, columns_to_datetime, columns_to_category, numeric_fill_na


@pytest.fixture(scope="module")
def df():
    return pd.DataFrame([
        {"a": 1, "b": "2008-03-03", "c": 0.0, "d1": np.nan, "d2": "x", "e": "h", "f": 5},
        {"a": 1, "b": "2008-03-04", "c": 1.0, "d1": "0"},
        {"a": 3, "b": "2008-03-05", "c": 3.0, "d1": "1", "f": 3},
        {"a": 3, "b": "2008-03-06", "c": 3.0, "d1": "1"},
        {"a": np.nan, "b": "2008-03-07", "c": 4.0, "d1": "2", "d2": "y", "e": "p"},
    ])


def test_regex_dict():
    rd = RegexDict({"^d1$": 1, "^d2$": 3})
    with pytest.raises(KeyError):
        print(rd["z3"])


def test_columns_to_datetime(df):
    with pytest.raises(TypeError):
        columns_to_datetime(df, "b", datetime_format=1)


def test_category2codes(df):
    with pytest.raises(TypeError):
        columns_to_category(df, ["d2"], columns_to_codes=10, to_codes=True)
    with pytest.raises(TypeError):
        columns_to_category(df, ["d2"], columns_to_codes=["d2"], to_codes=True, offset="mean")


def test_numeric_fill_na(df):
    with pytest.raises(TypeError, match="Cannot handle .* type offset"):
        numeric_fill_na(df, ["c"], mode=10)

    with pytest.raises(AttributeError, match=".*"):
        numeric_fill_na(df, ["a"], mode="error")

    with pytest.raises(ValueError, match="Cannot handle errors mode .*"):
        numeric_fill_na(df, ["a"], mode="error", errors="unknown")


def test_auto_df(df):
    df.info()
    auto_fti(
        df,
        category_columns=["d1"],
        columns_to_code=["$regex:d", "e"],
        category_code_offset=RegexDict({"^d1$": 1, "^d2$": 3}, default_value=5),
        datetime_columns=["b"],
        na_fill_mode="mode",
        verbose=True
    )
    df.info()
    print(df)
