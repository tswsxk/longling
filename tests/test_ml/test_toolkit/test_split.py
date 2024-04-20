# coding: utf-8
# 2020/4/17 @ tongshiwei

from pathlib import PurePath
import pytest
import json
from longling import as_out_io, loading, json_load, path_append
from longling.ml.toolkit.dataset import train_valid_test, train_test, kfold
from longling.ml.toolkit.dataset.split import get_s_indices


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


def test_utils(tmpdir, xy_path):
    with pytest.raises(ValueError):
        get_s_indices(*xy_path)

    index_file = path_append(tmpdir, "index.json")
    train_index, test_index = get_s_indices(*xy_path, ratio="1:1", index_file=index_file, shuffle=False)
    train_index, test_index = get_s_indices(*xy_path, s_indices=[train_index, test_index])
    assert list(train_index) == [0, 1, 2, 3, 4]
    assert list(test_index) == [5, 6, 7, 8, 9]

    train_index, test_index = json_load(index_file)

    assert train_index == [0, 1, 2, 3, 4]
    assert test_index == [5, 6, 7, 8, 9]

    train_index, test_index = get_s_indices(*xy_path, s_indices=index_file)
    assert list(train_index) == [0, 1, 2, 3, 4]
    assert list(test_index) == [5, 6, 7, 8, 9]


@pytest.mark.parametrize("ratio", [None, "8:2", (8, 2)])
def test_train_test(tmp_path, xy_path, ratio):
    x_path, y_path = xy_path
    if isinstance(ratio, tuple):
        train_test(x_path, y_path, train_size=ratio[0], test_size=ratio[1], prefix=str(tmp_path) + "/")
    else:
        train_test(x_path, y_path, prefix=str(tmp_path) + "/", ratio=ratio)
    for x, y in zip(loading(tmp_path / "x.train.jsonl"), loading(tmp_path / "y.train.txt", src_type="text")):
        assert x["z"] == int(y.strip())

    assert len(list(loading(tmp_path / "x.test.jsonl"))) == 2


@pytest.mark.parametrize("ratio", [None, "8:1:1", (8, 1, 1)])
def test_train_valid_test(tmp_path, xy_path, ratio):
    x_path, y_path = xy_path
    if isinstance(ratio, tuple):
        train_valid_test(x_path, y_path, train_size=ratio[0], valid_size=ratio[1], test_size=ratio[2],
                         prefix=str(tmp_path) + "/")
    else:
        train_valid_test(x_path, y_path, prefix=str(tmp_path) + "/", ratio=ratio)
    for x, y in zip(loading(tmp_path / "x.train.jsonl"), loading(tmp_path / "y.train.txt", src_type="text")):
        assert x["z"] == int(y.strip())

    assert len(list(loading(tmp_path / "x.valid.jsonl"))) == 1
    assert len(list(loading(tmp_path / "x.test.jsonl"))) == 1


def test_kfold(tmp_path, xy_path):
    x_path, y_path = xy_path

    kfold(x_path, y_path, prefix=str(tmp_path) + "/")

    for x, y in zip(loading(tmp_path / "x.0.train.jsonl"), loading(tmp_path / "y.0.train.txt")):
        assert x["z"] == int(y.strip())

    assert len(list(loading(tmp_path / "x.0.test.jsonl"))) == 2

    kfold(x_path, y_path)

    x_dir = PurePath(x_path).parent
    y_dir = PurePath(y_path).parent

    for x, y in zip(loading(x_dir / "x.0.train.jsonl"), loading(y_dir / "y.0.train.txt")):
        assert x["z"] == int(y.strip())

    assert len(list(loading(x_dir / "x.0.test.jsonl"))) == 2
