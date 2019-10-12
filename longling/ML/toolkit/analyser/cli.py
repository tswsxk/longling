# coding: utf-8
# 2019/10/12 @ tongshiwei

import json
from longling.lib.candylib import as_list


def key_parser(key):
    if "_" in key:
        # prf_0_f1
        return key.split("_")
    return key


def _select_max(src, *keys):
    keys = as_list(keys)

    result = {
        key: None for key in keys
    }

    with open(src) as f:
        for line in f:
            data = json.loads(line)
            for key in result:
                _key = key_parser(key)
                _data = data
                if isinstance(_key, list):
                    for k in _key:
                        _data = _data[k]
                else:
                    _data = _data[_key]
                if result[key] is None or _data > result[key]:
                    result[key] = _data
    for item in result.items():
        print(item)


def arg_select_max(*keys, src):
    _select_max(src, *keys)


def select_max(src, *keys):
    _select_max(src, *keys)


if __name__ == '__main__':
    arg_select_max("auc", "prf_1_f1", src="result.json")
