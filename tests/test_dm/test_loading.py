# coding: utf-8
# create by tongshiwei on 2020-10-28

import pandas as pd

from longling.dm.pdh import parallel_read_csv


def test_loading(tmp_path):
    a = [[1] * 10 for _ in range(3)]
    b = [[1] * 4 for _ in range(30)]

    df_a = pd.DataFrame(a)
    df_b = pd.DataFrame(b)

    df_a.to_csv(tmp_path / "a.csv")
    df_b.to_csv(tmp_path / "b.csv")

    a, b = parallel_read_csv([tmp_path / "a.csv", tmp_path / "b.csv"])
    assert len(a) == 3
    assert len(b) == 30

    df_a.to_csv(tmp_path / "a.csv", sep="\t")
    df_b.to_csv(tmp_path / "b.csv")

    a, b = parallel_read_csv([tmp_path / "a.csv", tmp_path / "b.csv"], params_group=[{"sep": "\t"}, None])
    assert len(a) == 3
    assert len(b) == 30
