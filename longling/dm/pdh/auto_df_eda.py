# coding: utf-8
# 2020/8/25 @ tongshiwei

import matplotlib.pyplot as plt
import pandas as pd

__all__ = ["plot_numeric", "quick_glance"]


def quick_glance(df: pd.DataFrame, plt_show=False, *args, **kwargs):
    df.info()

    plot_numeric(df, plt_show=plt_show, *args, **kwargs)

    print(df.head())


def plot_numeric(df: pd.DataFrame, plt_show=False, include: list = None, max_column_num=16):
    include = include if include else ['float64', 'int64']
    if len(df.columns) > max_column_num:
        df = df.iloc[:, : max_column_num]
    df_num = df.select_dtypes(include=include)
    df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

    if plt_show:  # pragma: no cover
        plt.show()
