from inspect import signature
from functools import wraps

from longling.base import LONGLING_TYPE_CHECK

__all__ = ['as_list', 'type_assert']


def as_list(obj):
    r"""A utility function that converts the argument to a list if it is not already.

    Parameters
    ----------
    obj : object
        argument to be converted to a list

    Returns
    -------
    list_obj: list
        If `obj` is a list or tuple, return it. Otherwise, return `[obj]` as a single-element list.

    Examples
    --------
    >>> as_list(1)
    [1]
    >>> as_list([1])
    [1]
    """
    if isinstance(obj, (list, tuple)):
        return obj
    else:
        return [obj]


def _type_assert(check_tag=True, *ty_args, **ty_kwargs):
    """

    Parameters
    ----------
    check_tag
    ty_args
    ty_kwargs

    Returns
    -------

    Examples
    --------
    >>> @type_assert(False, int, z=int)
    ... def spam(x, y, z=42):
    ...     print(x, y, z)
    ...
    >>> spam(1, 'hello', 3)
    1 hello 3
    >>> @type_assert(True, z=int)
    ... def spam(x, y, z=42):
    ...     print(x, y, z)
    ...
    >>> spam(1, 2, 3)
    1 2 3
    """

    def decorate(func):
        # Map function argument names to supplied types
        sig = signature(func)
        bound_types = sig.bind_partial(*ty_args, **ty_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            if check_tag:
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


def type_assert(*ty_args, **ty_kwargs):
    """

    Parameters
    ----------
    ty_args
    ty_kwargs

    Returns
    -------

    Examples
    --------
    >>> @type_assert(int, z=int)
    ... def spam(x, y, z=42):
    ...     print(x, y, z)
    ...
    >>> spam(1, 'hello', 3)
    1 hello 3
    >>> @type_assert(z=int)
    ... def spam(x, y, z=42):
    ...     print(x, y, z)
    ...
    >>> spam(1, 2, 3)
    1 2 3

    """
    return _type_assert(LONGLING_TYPE_CHECK, *ty_args, **ty_kwargs)


def get_all_subclass(cls):
    subclass = set()

    def _get_all_subclass(cls, res_set):
        res_set.add(cls)
        for sub_cls in cls.__subclasses__():
            _get_all_subclass(sub_cls, res_set)

    _get_all_subclass(cls, res_set=subclass)
    return subclass


class RegisterNameExistedError(Exception):
    pass


class Register(object):
    def __init__(self):
        self.reg = set()

    def register(self, name):
        if name not in self.reg:
            self.reg.add(name)
            return True
        else:
            raise RegisterNameExistedError("name %s existed" % name)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
