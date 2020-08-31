# coding: utf-8
# 2020/8/25 @ tongshiwei

import pandas as pd
from longling import as_list

__all__ = ["auto_fti", "auto_types", "category2codes", "columns_to_category", "numeric_fill_na"]


def auto_fti(df: pd.DataFrame,
             type_inference=True,
             category_columns: list = None, category_columns_regex=None,
             category_to_codes=True, columns_to_code: (str, list) = None, columns_not_to_code: (str, list) = None,
             category_code_offset=1,
             datetime_columns: (str, list) = None, datetime_format: (str, list) = None, datetime_kwargs: dict = None,
             **kwargs):
    """
    automatically feature typing and imputation

    Parameters
    ----------
    df
    type_inference
    category_columns
    category_columns_regex
    category_to_codes
    columns_to_code
    columns_not_to_code
    category_code_offset
    datetime_columns
    datetime_format

    kwargs

    Returns
    -------

    """
    category_columns = category_columns if category_columns else []

    if type_inference:
        auto_types(df, excluded=datetime_columns)

    if category_columns:
        columns_to_category(
            df, columns=category_columns, to_codes=False,
            regex=category_columns_regex,
            **kwargs.get("columns_to_category_kwargs", {})
        )

    if category_to_codes:
        category2codes(df, **kwargs.get("category2codes_kwargs", {}))

    if datetime_columns is not None:
        datetime_kwargs = datetime_kwargs if datetime_kwargs else {}
        columns_to_datetime(df, columns=datetime_columns, datetime_format=datetime_format, **datetime_kwargs)

    numeric_fill_na(df, **kwargs.get("numeric_fill_na", {}))

    return df


def auto_types(df: pd.DataFrame, excluded: (str, list) = None):
    excluded = set(as_list(excluded))
    for column in df.columns:
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


def columns_to_datetime(df: pd.DataFrame, columns: (str, list), datetime_format: (str, list) = None, regex=None, *args,
                        **kwargs):
    """

    Parameters
    ----------
    df
    columns
    datetime_format
    regex

    Returns
    -------

    """
    columns = as_list(columns)
    datetime_format = datetime_format if isinstance(datetime_format, list) else [datetime_format] * len(columns)

    for column, datetime_format in zip(columns, datetime_format):
        df[column] = pd.to_datetime(df[column], format=datetime_format, errors="coerce", *args, **kwargs)

    return df


def columns_to_category(df: pd.DataFrame,
                        columns: list, to_codes: bool = False, columns_not_to_codes: list = None, regex=None,
                        *args, **kwargs):
    """
    transfer the specified columns into category type

    Parameters
    ----------
    df
    columns
    to_codes
    columns_not_to_codes
    regex
    args
    kwargs

    Returns
    -------

    """
    if regex:
        fdf = df.filter(regex=regex)
        if not fdf.empty:
            df[fdf.columns] = fdf.astype("category")

    for column in columns:
        df[column] = df[column].astype(
            "category",
        )
    if to_codes:
        category2codes(df, columns=columns, ignore_columns=columns_not_to_codes, *args, **kwargs)

    return df


def category2codes(df: pd.DataFrame, offset: int = 1, columns: list = None, ignore_columns: (list, set) = None):
    """
    numerically encoding the categorical columns

    Parameters
    ----------
    df: pd.DataFrame
    offset: int
        0 or 1, default to 1 for the situation where exception or nan exists.
    columns: list
        categorical column to transfer to codes
    ignore_columns: list or set

    Returns
    -------

    """
    columns = columns if columns else df.select_dtypes(include=["category"]).columns
    ignore_columns = set(ignore_columns) if ignore_columns else set()
    for column in columns:
        if column not in ignore_columns:
            df[column] = df[column].cat.codes + offset

    return df


def numeric_fill_na(df: pd.DataFrame, columns: list = None, mode="mean", *args, **kwargs):
    columns = columns if columns else df.select_dtypes(include=['float64', 'int64']).columns
    for column in columns:
        fill_value = getattr(df[column], mode)(*args, **kwargs)
        df[column].fillna(fill_value, inplace=True)
    return df
