# coding: utf-8
# 2019/10/12 @ tongshiwei

import json
from longling.lib.candylib import as_list
from .select import get_max

__all__ = ["select_max", "arg_select_max"]


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
    result, result_appendix = get_max(src, *keys, with_keys=with_keys, with_all=with_all)

    for item in result.items():
        print(item, end='')
        if result_appendix[item[0]]:
            print(" ", result_appendix[item[0]])
        else:
            print()


def arg_select_max(*keys, src, with_keys=None, with_all=False):
    """
    cli alias: ``amax``

    Parameters
    ----------
    keys
    src
    with_keys
    with_all

    Returns
    -------

    """
    _select_max(src, *keys, with_keys=with_keys, with_all=with_all)


def select_max(src, *keys, with_keys=None, with_all=False):
    """
    cli alias: ``max``

    Parameters
    ----------
    src
    keys
    with_keys
    with_all

    Returns
    -------

    """
    _select_max(src, *keys, with_keys=with_keys, with_all=with_all)
