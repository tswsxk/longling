from inspect import signature
from functools import wraps


def type_assert(check_tag=True, *ty_args, **ty_kwargs):
    '''
    :param ty_args:
    :param ty_kwargs:
    :return:
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
    '''

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


if __name__ == '__main__':
    import doctest

    doctest.testmod()
