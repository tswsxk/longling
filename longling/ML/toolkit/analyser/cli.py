# coding: utf-8
# 2019/10/12 @ tongshiwei

import json
from longling.lib.candylib import as_list


def key_parser(key):
    if ":" in key:
        # prf:0:f1
        return key.split(":")
    return key


def get_by_key(data, parsed_key):
    _key = key_parser(parsed_key)
    _data = data
    if isinstance(_key, list):
        for k in _key:
            _data = _data[k]
    else:
        _data = _data[_key]
    return _data


def _select_max(src, *keys, with_keys: (str, None) = None, with_all=False):
    keys = as_list(keys)

    with_keys = [] if with_keys is None else with_keys.split(";")

    result = {
        key: None for key in keys
    }

    result_appendix = {
        key: None for key in keys
    }

    with open(src) as f:
        for line in f:
            data = json.loads(line)
            for key in result:
                _data = get_by_key(data, parsed_key=key_parser(key))
                if result[key] is None or _data > result[key]:
                    result[key] = _data
                    if with_all:
                        result_appendix[key] = data
                    elif with_keys:
                        result_appendix[key] = [(_key, get_by_key(data, key_parser(_key))) for _key in with_keys]

    for item in result.items():
        print(item, end='')
        if result_appendix[item[0]]:
            print(" ", result_appendix[item[0]])
        else:
            print()


def arg_select_max(*keys, src, with_keys=None, with_all=False):
    _select_max(src, *keys, with_keys=with_keys, with_all=with_all)


def select_max(src, *keys, with_keys=None, with_all=False):
    _select_max(src, *keys, with_keys=with_keys, with_all=with_all)
