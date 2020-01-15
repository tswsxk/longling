# coding: utf-8
# 2019/12/20 @ tongshiwei

import json
from longling.lib.candylib import as_list
from longling import PATH_TYPE, loading

__all__ = ["get_max", "key_parser", "get_by_key"]


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


def get_max(src: (PATH_TYPE, list), *keys, with_keys: (str, None) = None, with_all=False):
    keys = as_list(keys)

    with_keys = [] if with_keys is None else with_keys.split(";")

    result = {
        key: None for key in keys
    }

    result_appendix = {
        key: None for key in keys
    }

    for data in loading(src):
        for key in result:
            _data = get_by_key(data, parsed_key=key_parser(key))
            if result[key] is None or _data > result[key]:
                result[key] = _data
                if with_all:
                    result_appendix[key] = data
                elif with_keys:
                    result_appendix[key] = [(_key, get_by_key(data, key_parser(_key))) for _key in with_keys]

    return result, result_appendix
