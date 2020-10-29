# coding: utf-8
# create by tongshiwei on 2020-10-29

__all__ = ["group_sort"]

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
