# coding: utf-8
# 2019/10/12 @ tongshiwei

from longling.lib.formatter import pandas_format, dict_format
from .select import get_max, get_min

__all__ = ["select_max", "arg_select_max"]


def _select(src, *keys, with_keys: (str, None) = None, with_all=False, func=None, **kwargs):
    res = func(src, *keys, with_keys=with_keys, with_all=with_all)
    if with_keys or with_all:
        print(pandas_format(res, **kwargs))
    else:
        print(dict_format(res, **kwargs))


def _select_max(src, *keys, with_keys: (str, None) = None, with_all=False, **kwargs):
    _select(src, *keys, with_keys=with_keys, with_all=with_all, func=get_max, **kwargs)


def _select_min(src, *keys, with_keys: (str, None) = None, with_all=False, **kwargs):
    _select(src, *keys, with_keys=with_keys, with_all=with_all, func=get_min, **kwargs)


def arg_select_max(*keys, src, with_keys=None, with_all=False, **kwargs):
    """
    cli alias: ``amax``
    """
    _select_max(src, *keys, with_keys=with_keys, with_all=with_all, **kwargs)


def select_max(src, *keys, with_keys=None, with_all=False, **kwargs):
    """
    cli alias: ``max``
    """
    _select_max(src, *keys, with_keys=with_keys, with_all=with_all, **kwargs)


def arg_select_min(*keys, src, with_keys=None, with_all=False, **kwargs):
    """
    cli alias: ``amin``
    """
    _select_min(src, *keys, with_keys=with_keys, with_all=with_all, **kwargs)


def select_min(src, *keys, with_keys=None, with_all=False, **kwargs):
    """
    cli alias: ``min``
    """
    _select_min(src, *keys, with_keys=with_keys, with_all=with_all, **kwargs)
