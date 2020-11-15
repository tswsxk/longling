# coding: utf-8
# create by tongshiwei on 2020-10-29

__all__ = ["group_sort", "flatten_df"]

import itertools

import pandas as pd


def group_sort(df: pd.DataFrame, group_by, sort_by,
               group_params=None, sort_params=None,
               reset_index=True):
    """

    Parameters
    ----------
    df
    group_by
    sort_by

    Returns
    -------

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {"id": [1, 2, 3, 1, 1, 2], "idx": [0, 1, 0, 2, 1, 1], "course": ["cs", "math", "bio", "cs", "math", "math"]
    ... })
    >>> group_sort(df, "id", "idx")
       id  idx course
    0   1    0     cs
    1   1    1   math
    2   1    2     cs
    3   2    1   math
    4   2    1   math
    5   3    0    bio
    """
    df = df.groupby(
        by=group_by, sort=False, **({} if group_params is None else group_params)
    ).apply(
        lambda x: x.sort_values(sort_by, **({} if sort_params is None else sort_params))
    )
    if reset_index:
        return df.reset_index(drop=True)
    else:  # pragma: no cover
        return df


def flatten_df(df: pd.DataFrame):
    """
    Examples
    -------
    >>> import pandas as pd
    >>> _df = pd.DataFrame({"a": [1, 2, 3], "b": [0, 1, 2]})
    >>> flatten_df(_df)
       a_0  b_0  a_1  b_1  a_2  b_2
    0    1    0    2    1    3    2
    """
    num = len(df)
    columns = df.columns
    new_columns = list(itertools.chain(*[[column + "_%s" % i for column in columns] for i in range(num)]))
    flatten_value = df.values.flatten().tolist()
    return pd.DataFrame([flatten_value], columns=new_columns)
