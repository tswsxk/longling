# coding: utf-8
# create by tongshiwei on 2017/12/2

import sys

from inspect import signature
from functools import wraps

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


def typeassert(*ty_args, **ty_kwargs):
    '''
    :param ty_args:
    :param ty_kwargs:
    :return:
    >>> @typeassert(int, z=int)
    ... def spam(x, y, z=42):
    ...     print(x, y, z)
    ...
    >>> spam(1, 'hello', 3)
    1 hello 3
    >>> @typeassert(z=int)
    ... def spam(x, y, z=42):
    ...     print(x, y, z)
    ...
    >>> spam(1, 2, 3)
    1 2 3
    '''

    def decorate(func):
        # Map function argument names to supplied types
        sig = signature(func)
        bound_types = sig.bind_partial(*ty_args, **ty_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            # Enforce type assertions across supplied arguments
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError(
                            'Argument {} must be {}'.format(name, bound_types[name])
                        )
            return func(*args, **kwargs)

        return wrapper

    return decorate
