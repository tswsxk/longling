# coding: utf-8
# 2020/3/27 @ tongshiwei

from collections import OrderedDict
from longling.utils.stream import as_out_io
from longling.utils.path import path_append
from longling.utils.yaml_helper import dump_folded_yaml, ordered_yaml_load, FoldedString


def test_yaml(tmp_path):
    tmp_file = path_append(tmp_path, "test.yaml")

    obj = OrderedDict(
        {
            "b": 123,
        }
    )
    obj["a"] = 456

    c = ""
    c += "helm install" + "\n"
    c += "--set \"abc\"" + "\n"

    obj["c"] = [FoldedString(c)]

    with as_out_io(tmp_file) as wf:
        print(dump_folded_yaml(obj), file=wf)

    for i, (key, value) in enumerate(ordered_yaml_load(tmp_file).items()):
        if i == 0:
            assert key == "b"
            assert value == 123
        elif i == 1:
            assert key == "a"
            assert value == 456
        elif i == 2:
            assert key == "c"
            assert value == ['helm install --set "abc"\n']
