# coding: utf-8
# create by tongshiwei on 2020-10-12

import pandas as pd

from longling.dm import quick_glance, plot_numeric


def test_auto_df_eda():
    df = pd.DataFrame({
        "a": range(10),
        "b": range(1, 11),
    })

    quick_glance(df)

    plot_numeric(df, max_column_num=1)
