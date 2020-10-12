# coding: utf-8
# create by tongshiwei on 2020-10-12

import pandas as pd

from longling.DM import quick_glance


def test_auto_df_eda():
    df = pd.DataFrame({
        "a": range(10),
        "b": range(1, 11),
    })

    quick_glance(df)
