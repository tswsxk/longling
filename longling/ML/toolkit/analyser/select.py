# coding: utf-8
# 2019/12/20 @ tongshiwei

from collections import OrderedDict
from longling.lib.candylib import as_list
from longling import PATH_TYPE, loading

__all__ = ["get_max", "get_by_key", "get_best", "get_min"]

from .utils import get_by_key


def get_best(src: (PATH_TYPE, list), *keys, with_keys: (str, None) = None, with_all=False, cmp=lambda x, y: x > y,
             merge=True):
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
            _data = get_by_key(data, parsed_key=key)
            if result[key] is None or cmp(_data, result[key]):
                result[key] = _data
                if with_all:
                    result_appendix[key] = data
                elif with_keys:
                    result_appendix[key] = {_key: get_by_key(data, _key) for _key in with_keys}

    if merge:
        return _merge(result, result_appendix if with_all or with_keys else None)
    else:
        return result, result_appendix if with_all or with_keys else None


def get_max(src: (PATH_TYPE, list), *keys, with_keys: (str, None) = None, with_all=False, merge=True):
    """
    Examples
    -------
    >>> src = [
    ... {"Epoch": 0, "macro avg": {"f1": 0.7}, "loss": 0.04, "accuracy": 0.7},
    ... {"Epoch": 1, "macro avg": {"f1": 0.88}, "loss": 0.03, "accuracy": 0.8},
    ... {"Epoch": 1, "macro avg": {"f1": 0.7}, "loss": 0.02, "accuracy": 0.66}
    ... ]
    >>> result, _ = get_max(src, "accuracy", merge=False)
    >>> result
    {'accuracy': 0.8}
    >>> _, result_appendix = get_max(src, "accuracy", with_all=True, merge=False)
    >>> result_appendix
    {'accuracy': {'Epoch': 1, 'macro avg': {'f1': 0.88}, 'loss': 0.03, 'accuracy': 0.8}}
    >>> result, result_appendix = get_max(src, "accuracy", "macro avg:f1", with_keys="Epoch", merge=False)
    >>> result
    {'accuracy': 0.8, 'macro avg:f1': 0.88}
    >>> result_appendix
    {'accuracy': {'Epoch': 1}, 'macro avg:f1': {'Epoch': 1}}
    >>> get_max(src, "accuracy", "macro avg:f1", with_keys="Epoch")
    {'accuracy': {'Epoch': 1, 'accuracy': 0.8}, 'macro avg:f1': {'Epoch': 1, 'macro avg:f1': 0.88}}
    """
    return get_best(
        src, *keys, with_keys=with_keys, with_all=with_all, merge=merge
    )


def get_min(src: (PATH_TYPE, list), *keys, with_keys: (str, None) = None, with_all=False, merge=True):
    """
    >>> src = [
    ... {"Epoch": 0, "macro avg": {"f1": 0.7}, "loss": 0.04, "accuracy": 0.7},
    ... {"Epoch": 1, "macro avg": {"f1": 0.88}, "loss": 0.03, "accuracy": 0.8},
    ... {"Epoch": 1, "macro avg": {"f1": 0.7}, "loss": 0.02, "accuracy": 0.66}
    ... ]
    >>> get_min(src, "loss")
    {'loss': 0.02}
    """
    return get_best(
        src, *keys, with_keys=with_keys, with_all=with_all, cmp=lambda x, y: x < y, merge=merge
    )


def _merge(result: dict, appendix):
    """
    Examples
    --------
    >>> _merge({"accuracy": 0.7}, {"accuracy": {"Epoch": 1, "accuracy": 0.7}})
    {'accuracy': {'Epoch': 1, 'accuracy': 0.7}}
    """
    if appendix:
        for key in appendix:
            appendix[key][key] = result[key]
        return appendix
    else:
        return result


if __name__ == '__main__':
    src = [
        {"Epoch": 0, "macro avg": {"f1": 0.7}, "loss": 0.04, "accuracy": 0.7},
        {"Epoch": 1, "macro avg": {"f1": 0.88}, "loss": 0.03, "accuracy": 0.8},
        {"Epoch": 1, "macro avg": {"f1": 0.7}, "loss": 0.02, "accuracy": 0.66}
    ]
    print(get_max(src, "accuracy", "macro avg:f1", with_keys="Epoch", merge=False))
