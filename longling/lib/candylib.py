__all__ = ['as_list']


def as_list(obj):
    r"""A utility function that converts the argument to a list
    if it is not already.

    Parameters
    ----------
    obj : object
        argument to be converted to a list

    Returns
    -------
    list_obj: list
        If `obj` is a list or tuple, return it. Otherwise,
        return `[obj]` as a single-element list.

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
