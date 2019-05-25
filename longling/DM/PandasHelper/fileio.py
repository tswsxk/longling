# coding: utf-8
# create by tongshiwei on 2019/5/25
"""
Some io methods as the supplement for some interface of pandas.
For example:

* read_csv: extend the functionality and fix bugs in pandas.read_csv

"""

import csv

import pandas as pd
from tqdm import tqdm


def read_csv(csv_fp, skip_header=False, silent=True, **kwargs):
    reader = iter(tqdm(csv.reader(csv_fp, **kwargs), disable=not silent))
    header = None if skip_header else next(reader)
    body = reader
    return pd.DataFrame(list(body), columns=header)
