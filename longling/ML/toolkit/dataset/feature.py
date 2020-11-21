# coding: utf-8
# create by tongshiwei on 2020-11-21

import pandas as pd

__all__ = ["ID2Feature"]


class ID2Feature(object):
    """
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"id": [0, 1, 2, 3, 4], "numeric": [1, 2, 3, 4, 5], "text": ["a", "b", "c", "d", "e"]})
    >>> i2f = ID2Feature(df, id_field="id", set_index=True)
    >>> i2f[2]
    numeric    3
    text       c
    Name: 2, dtype: object
    >>> i2f[[2, 3]]["numeric"]
    id
    2    3
    3    4
    Name: numeric, dtype: int64
    >>> i2f(2)
    [3, 'c']
    >>> i2f([2, 3])
    [[3, 'c'], [4, 'd']]
    """

    def __init__(self, feature_df: pd.DataFrame, id_field=None, set_index=False):
        self.feature_df = feature_df
        if set_index:
            self.feature_df.set_index(id_field, inplace=True)

    def __call__(self, id):
        return self[id].values.tolist()

    def __getitem__(self, item):
        return self.feature_df.iloc[item]
