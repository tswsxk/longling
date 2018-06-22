# coding: utf-8
# create by tongshiwei on 2017/12/2
from __future__ import absolute_import

import sys

from longling.lib.candylib import type_assert as _typeassert

if sys.version_info[0] == 3:
    string_types = str,
    integer_types = int
    # this function is needed for python3
    # to convert ctypes.char_p .value back to python str
    unistr = lambda x: x
    tostr = lambda x: str(x)
else:
    string_types = basestring,
    integer_types = (int, long)
    unistr = lambda x: x.decode('utf-8')
    tostr = lambda x: unicode(x)

LONGLING_TYPE_CHECK = True


def type_assert(*ty_args, **ty_kwargs):
    return _typeassert(LONGLING_TYPE_CHECK, *ty_args, **ty_kwargs)
