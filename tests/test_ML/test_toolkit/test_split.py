# coding: utf-8
# 2020/4/17 @ tongshiwei

import pytest
import json
from longling import as_out_io, loading
from longling.ML.toolkit.dataset import train_valid_test, train_test, kfold


@pytest.fixture(scope="module")
def xy_path(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("data")
    _x_path = tmp_dir / "x.jsonl"
    _y_path = tmp_dir / "y.txt"
    with as_out_io(_x_path) as xf, as_out_io(_y_path) as yf:
        for i in range(10):
            print(json.dumps({"x": [j for j in range(i, i + 5)], "z": i}), file=xf)
            print(i, file=yf)

    return _x_path, _y_path


def test_train_test(tmp_path, xy_path):
    x_path, y_path = xy_path
    train_test(x_path, y_path, prefix=str(tmp_path) + "/")
    for x, y in zip(loading(tmp_path / "x.train.jsonl"), loading(tmp_path / "y.train.txt", src_type="text")):
        assert x["z"] == int(y.strip())


def test_train_valid_test(tmp_path, xy_path):
    x_path, y_path = xy_path
    train_valid_test(x_path, y_path, prefix=str(tmp_path) + "/")
    for x, y in zip(loading(tmp_path / "x.train.jsonl"), loading(tmp_path / "y.train.txt", src_type="text")):
        assert x["z"] == int(y.strip())


def test_kfold(tmp_path, xy_path):
    x_path, y_path = xy_path

    kfold(x_path, y_path, prefix=str(tmp_path) + "/")
    for x, y in zip(loading(tmp_path / "x.0.train.jsonl"), loading(tmp_path / "y.0.train.txt")):
        assert x["z"] == int(y.strip())
