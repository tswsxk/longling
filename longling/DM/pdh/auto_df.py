# coding: utf-8
# 2020/8/25 @ tongshiwei

import functools
import logging
import re
from collections import Iterable

import pandas as pd
from tqdm import tqdm

from longling import as_list

__all__ = ["auto_fti", "auto_types", "category2codes", "columns_to_category", "numeric_fill_na", "RegexDict"]


def _log(msg, *args, logger=logging, verbose=False, level=logging.INFO, **kwargs):
    if verbose:
        logger.log(level, msg, *args, **kwargs)


def _get_log_f(verbose=False, logger=logging, level=logging.INFO, **kwargs):
    return functools.partial(_log, logger=logger, verbose=verbose, level=level, **kwargs)


class RegexDict(dict):
    """
    Examples
    --------
    >>> rd = RegexDict({"[a-z]1$": 1, "[a-z]2$": 3}, default_value=5)
    >>> rd["x1"]
    1
    >>> rd["y1"]
    1
    >>> rd["z2"]
    3
    >>> rd["z3"]
    5
    """
    def __init__(self, *args, default_value=None, **kwargs):
        super(RegexDict, self).__init__(*args, **kwargs)
        self._default_value = default_value
        self._pattern = re.compile(
            r"(?P<pattern_str>.*)"
        )
        self._pattern_value_dict = []
        for key, value in self.items():
            _match = self._pattern.match(key)

            _pattern_str = _match.group("pattern_str")
            self._pattern_value_dict.append((re.compile(_pattern_str), value))

    def __getitem__(self, item):
        for pattern, value in self._pattern_value_dict:
            if pattern.findall(item):
                return value
        if self._default_value is not None:
            return self._default_value
        raise KeyError(item)


class _Regex(object):
    """
    Examples
    --------
    >>> exclude_regex = _Regex()
    >>> exclude_regex.add("[!]abc")
    >>> exclude_regex.add("[!]$regex:id")
    >>> exclude_regex(["abc", "subject_id", "bbc"])
    ['bbc']
    >>> include_regex = _Regex()
    >>> include_regex.add("abc", "$regex:id")
    >>> include_regex(["abc", "subject_id", "bbc"])
    ['abc', 'subject_id']
    """

    def __init__(self, columns: Iterable = None):
        self._pattern = re.compile(
            r"^(?P<exclude_tag>\[!\])*(?P<regex_tag>\$regex:)*(?P<pattern_str>.*)"
        )
        self.exact_include_columns = set()
        self.exact_exclude_columns = set()
        self.regex_include_columns = []
        self.regex_exclude_columns = []
        self._mode = None
        if columns is not None:
            self.add(*as_list(columns))

    def add(self, *columns):
        for column in as_list(columns):
            if column is None:  # pragma: no cover
                continue

            _match = self._pattern.match(column)

            _exclude_tag = _match.group("exclude_tag")
            _regex_tag = _match.group("regex_tag")
            _pattern_str = _match.group("pattern_str")

            if not _regex_tag:
                if _exclude_tag:
                    self.exact_exclude_columns.add(_pattern_str)
                else:
                    self.exact_include_columns.add(_pattern_str)
            else:
                if _exclude_tag:
                    self.regex_exclude_columns.append(re.compile(_pattern_str))
                else:
                    self.regex_include_columns.append(re.compile(_pattern_str))
        assert not (
                (self.exact_include_columns or self.regex_include_columns
                 ) and (self.exact_exclude_columns or self.regex_exclude_columns)
        ), "include mode and exclude mode are exclusive"

        self._mode = "include" if self.exact_include_columns or self.regex_include_columns else "exclude"

    def __call__(self, columns: Iterable, verbose=False, *args, **kwargs):
        ret = []
        if self._mode == "include":
            for column in columns:
                column = str(column)
                if column in self.exact_include_columns:
                    ret.append(column)
                else:
                    for p in self.regex_include_columns:
                        if p.findall(column):
                            ret.append(column)
                            break
            return ret

        elif self._mode == "exclude":
            for column in columns:
                _exclude = False
                if column in self.exact_exclude_columns:
                    _exclude = True
                else:
                    for p in self.regex_exclude_columns:
                        if p.findall(column):
                            _exclude = True
                            break
                if not _exclude:
                    ret.append(column)
            return ret
        else:  # pragma: no cover
            raise ValueError("mode type should be either include or exclude")


def _as_regex(pattern: (str, _Regex, Iterable)):  # pragma: no cover
    if isinstance(pattern, _Regex):
        return pattern
    else:
        return _Regex(pattern)


def _filter_columns(pattern_columns: (str, Iterable, _Regex), to_filter: Iterable, verbose=False):
    return _as_regex(pattern_columns)(to_filter, verbose=verbose)


def _get_columns_by_dtype(df: pd.DataFrame, dtype: (str, list)):
    dtype = as_list(dtype)
    return df.select_dtypes(include=dtype).columns


def auto_types(df: pd.DataFrame, excluded: (str, Iterable) = None, verbose=False, pattern_mode=False, **kwargs):
    """
    Only infer the type of object

    Parameters
    ----------
    df
    excluded
    verbose
    pattern_mode: bool
        When pattern mode is set as True,
        matching columns will be inferred using regex pattern, which is time consuming

    Returns
    -------


    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [0.1, 0.2, 0.3, 0.4, 0.5], "c": ["a", "b", "c", "d", "e"]})
    >>> df = auto_types(df)
    >>> df.dtypes
    a       int64
    b     float64
    c    category
    dtype: object
    >>> df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [0.1, 0.2, 0.3, 0.4, 0.5], "c": ["a", "b", "c", "d", "e"]})
    >>> df = auto_types(df, excluded=["c"])
    >>> df.dtypes
    a      int64
    b    float64
    c     object
    dtype: object
    """
    __log = _get_log_f(verbose=verbose, **kwargs)
    if excluded:
        excluded = set(
            as_list(excluded) if not pattern_mode
            else _filter_columns([e for e in excluded], df.columns)
        )
    else:
        excluded = set()

    if excluded:
        __log("Auto typing: excluded columns: %s" % ", ".join(excluded))

    for column in tqdm(_get_columns_by_dtype(df, "object"), "auto typing", disable=not verbose):
        if column in excluded:
            continue
        numeric_column = pd.to_numeric(df[column].copy(), errors="coerce")
        if numeric_column.count() > 0:
            df[column] = numeric_column
        else:
            df[column] = df[column].astype(
                "category",
            )
    return df


def columns_to_datetime(
        df: pd.DataFrame,
        columns: (str, list),
        datetime_format: (str, list) = None,
        pattern_mode=False,
        verbose=False,
        *args, **kwargs):
    """

    Parameters
    ----------
    df
    columns: str or list
        The columns to be interpreted as datetime
    datetime_format
    pattern_mode: bool
        When pattern mode is set as True,
        matching columns will be inferred using regex pattern, which is time consuming
    verbose

    Returns
    -------

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"id": [1, 2, 3], "t": ["1931-09-18", "1949-10-01", "2020-10-12"]})
    >>> df = columns_to_datetime(df, "t")
    >>> df.dtypes
    id             int64
    t     datetime64[ns]
    dtype: object
    >>> df = pd.DataFrame({"t1": ["1931-09-18", "1949-10-01", "unknown"], "t2": ["20201012", "20201013", "unknown"]})
    >>> df = columns_to_datetime(df, ["t1", "t2"])
    >>> df.dtypes
    t1    datetime64[ns]
    t2    datetime64[ns]
    dtype: object
    >>> df
              t1         t2
    0 1931-09-18 2020-10-12
    1 1949-10-01 2020-10-13
    2        NaT        NaT
    >>> df = pd.DataFrame({
    ...     "t1": ["1931-09-18:023358", "1949-10-01:040909"],
    ...     "t2": ["20201012-011203", "20201013-070301"]
    ... })
    >>> df = columns_to_datetime(
    ...     df, ["t1", "t2"],
    ...     datetime_format=["%Y-%m-%d:%H%M%S", "%Y%m%d-%H%M%S"]
    ... )
    >>> df
                       t1                  t2
    0 1931-09-18 02:33:58 2020-10-12 01:12:03
    1 1949-10-01 04:09:09 2020-10-13 07:03:01
    >>> df = pd.DataFrame({
    ...     "t1": ["1931-09-18:023358", "1949-10-01:040909"],
    ...     "t2": ["20201012-011203", "20201013-070301"]
    ... })
    >>> df = columns_to_datetime(
    ...     df, ["t1", "t2"],
    ...     datetime_format={"t1": "%Y-%m-%d:%H%M%S", "t2": "%Y%m%d-%H%M%S"}
    ... )
    >>> df
                       t1                  t2
    0 1931-09-18 02:33:58 2020-10-12 01:12:03
    1 1949-10-01 04:09:09 2020-10-13 07:03:01
    """
    __log = _get_log_f(verbose=verbose, **kwargs)

    columns = as_list(columns)

    if datetime_format is None or isinstance(datetime_format, (str, list)):
        columns = _filter_columns(columns, df.columns) if pattern_mode is True else columns
        if isinstance(datetime_format, list):
            assert len(columns) == len(datetime_format), "columns[%s] vs datetime_format[%s]" % (
                len(columns), len(datetime_format)
            )
            __log("columns to be datetime: %s" % list(zip(columns, datetime_format)))
        else:
            __log("columns to be datetime: %s" % ", ".join(columns))
            __log("datetime format is %s" % datetime_format)
            datetime_format = [datetime_format] * len(columns)
    elif isinstance(datetime_format, dict):
        # assert pattern_mode is True, "set pattern mode as True when datetime_format is dict"
        columns = _filter_columns(columns, df.columns)
        datetime_format = [
            datetime_format[column] for column in columns
        ]
        __log("columns to be datetime: %s" % list(zip(columns, datetime_format)))
    else:
        raise TypeError("Cannot handle %s type datetime_format" % type(datetime_format))

    for column, datetime_format in tqdm(zip(columns, datetime_format), "columns to datetime", disable=not verbose):
        df[column] = pd.to_datetime(df[column], format=datetime_format, errors="coerce", *args, **kwargs)

    return df


def columns_to_category(df: pd.DataFrame,
                        columns: list, to_codes: bool = False, columns_to_codes: list = None,
                        pattern_mode=True,
                        code_pattern_mode=None,
                        verbose=False,
                        *args, **kwargs):
    """
    transfer the specified columns into category type

    Parameters
    ----------
    df
    columns
    to_codes
    columns_to_codes
    pattern_mode: bool
        When pattern mode is set as True,
        matching columns will be inferred using regex pattern, which is time consuming
    code_pattern_mode: bool or None
        When pattern mode is set as True,
        matching columns will be inferred using regex pattern, which is time consuming
    verbose
    args
    kwargs

    Returns
    -------

    """
    __log = _get_log_f(verbose=verbose, **kwargs)

    columns = as_list(columns)
    columns = _filter_columns(columns, df.columns) if pattern_mode else columns

    __log("columns to be categorical: %s" % columns)

    for column in tqdm(columns, "columns to be categorical", disable=not verbose):
        if df[column].dtype.name == "category":
            __log("column[%s] has been categorical, ignored" % column)
            continue
        df[column] = df[column].astype(
            "category",
        )
    if to_codes:
        code_pattern_mode = pattern_mode if code_pattern_mode is None else code_pattern_mode
        columns = _filter_columns(columns_to_codes, columns) if code_pattern_mode else as_list(columns_to_codes)
        category2codes(df, columns=columns, verbose=verbose, pattern_mode=False, **kwargs)

    return df


def category2codes(
        df: pd.DataFrame, offset: (int, list, dict) = 1, columns: (str, Iterable) = None,
        pattern_mode=True, verbose=False,
        **kwargs):
    """
    numerically encoding the categorical columns

    Parameters
    ----------
    df: pd.DataFrame
    offset: (int, list, dict)
        0 or 1, default to 1 for the situation where exception or nan exists.
    columns: str or Iterable
        categorical column to transfer to codes
    pattern_mode
    verbose

    Returns
    -------

    """
    __log = _get_log_f(verbose=verbose, **kwargs)

    columns = as_list(columns) if columns else df.select_dtypes(include=["category"]).columns.values.tolist()

    if isinstance(offset, (int, list)):
        columns = _filter_columns(columns, df.columns) if pattern_mode is True else columns
        if isinstance(offset, list):
            assert len(columns) == len(offset), "columns[%s] vs offset[%s]" % (
                len(columns), len(offset)
            )
            __log("categorical columns to be coded: %s" % list(zip(columns, offset)))
        else:
            __log("categorical columns to be coded: %s" % ", ".join(columns))
            __log("offset is %s" % offset)
            offset = [offset] * len(columns)
    elif isinstance(offset, dict):
        assert pattern_mode is True, "set pattern mode as True when offset is dict"
        columns = _filter_columns(columns, df.columns)
        offset = [
            offset[column] for column in columns
        ]
        __log("categorical columns to be coded: %s" % list(zip(columns, offset)))
    else:
        raise TypeError("Cannot handle %s type offset" % type(offset))

    for column, _offset in tqdm(zip(columns, offset), "encoding categorical columns", disable=not verbose):
        df[column] = df[column].cat.codes + _offset

    return df


def numeric_fill_na(
        df: pd.DataFrame, columns: (str, Iterable) = None,
        mode: (str, dict, Iterable) = "mean", pattern_mode=False, verbose=False,
        errors="raise",
        *args, **kwargs):
    """

    Parameters
    ----------
    df
    columns
    mode
    pattern_mode
    verbose
    errors
    args
    kwargs

    Returns
    -------

    """
    __log = _get_log_f(verbose=verbose, **kwargs)

    df_columns = df.select_dtypes(include=['float64', 'int64']).columns.values.tolist()
    columns = columns if columns is not None else df_columns

    if isinstance(mode, (str, list)):
        columns = _filter_columns(columns, df_columns) if pattern_mode is True else columns
        if isinstance(mode, list):
            assert len(columns) == len(mode), "columns[%s] vs offset[%s]" % (
                len(columns), len(mode)
            )
            __log("numerical columns to be filled nan: %s" % list(zip(columns, mode)))
        else:
            __log("numerical columns to be filled nan: %s" % ", ".join(columns))
            __log("mode is %s" % mode)
            mode = [mode] * len(columns)
    elif isinstance(mode, dict):
        assert pattern_mode is True, "set pattern mode as True when mode is dict"
        columns = _filter_columns(columns, df.columns)
        mode = [
            mode[column] for column in columns
        ]
        __log("numerical columns to be filled nan: %s" % list(zip(columns, mode)))
    else:
        raise TypeError("Cannot handle %s type offset" % type(mode))

    for column, _mode in tqdm(zip(columns, mode), "filling nan", disable=not verbose):
        if df[column].isnull().values.any():
            try:
                if _mode == "mode":
                    fill_value = df[column].mode().values[0]
                else:
                    fill_value = getattr(df[column], _mode)(*args, **kwargs)
                df[column].fillna(fill_value, inplace=True)
            except AttributeError as e:
                if errors == "zero":
                    df[column].fillna(0, inplace=True)
                elif errors == "raise":
                    raise e
                elif errors == "ignore":
                    __log("error in column %s, skipped" % column)
                else:
                    raise ValueError("Cannot handle errors mode %s" % errors)
        else:
            __log("no nan in column %s, skipped" % column)
    return df


def auto_fti(df: pd.DataFrame,
             type_inference=True,
             category_columns: (str, Iterable) = None,
             category_to_codes=True, columns_to_code: (str, Iterable) = None,
             category_code_offset: (int, list, dict) = 1,
             datetime_columns: (str, Iterable) = None, datetime_format: (str, Iterable) = None,
             auto_na_fill: bool = True, columns_to_fill_na: (str, Iterable) = None, na_fill_mode="mean",
             verbose=False,
             **kwargs):
    """
    automatically feature typing and imputation

    Parameters
    ----------
    df
    type_inference
    category_columns
    category_to_codes
    columns_to_code
    category_code_offset: int, list or dict
    datetime_columns
    datetime_format
    auto_na_fill
    columns_to_fill_na
    na_fill_mode
    verbose

    kwargs

    Returns
    -------


    Examples
    --------


    """
    __log = _get_log_f(verbose=verbose, **kwargs)

    if type_inference:
        auto_types(df, excluded=datetime_columns, verbose=verbose, pattern_mode=True, **kwargs)

    if category_columns:
        columns_to_category(
            df,
            columns=category_columns, to_codes=False,
            pattern_mode=True,
            verbose=verbose,
            **kwargs
        )

    if category_to_codes:
        category2codes(
            df,
            columns=columns_to_code,
            offset=category_code_offset,
            verbose=verbose,
            pattern_mode=True,
            **kwargs
        )

    if datetime_columns is not None:
        columns_to_datetime(
            df, columns=datetime_columns, datetime_format=datetime_format,
            pattern_mode=True,
            verbose=verbose,
            **kwargs
        )

    if auto_na_fill:
        numeric_fill_na(
            df,
            columns=columns_to_fill_na,
            mode=na_fill_mode,
            verbose=verbose,
            pattern_mode=True,
            errors="raise",
            **kwargs
        )

    return df
