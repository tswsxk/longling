__all__ = ['as_list', 'dict2pv', 'list2dict', 'get_dict_by_path']


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


def list2dict(list_obj, value=None, dict_obj=None):
    """
    >>> list_obj = ["a", 2, "c"]
    >>> list2dict(list_obj, 10)
    {'a': {2: {'c': 10}}}
    """
    dict_obj = {} if dict_obj is None else dict_obj
    _dict_obj = dict_obj
    for e in list_obj[:-1]:
        if e not in _dict_obj:
            _dict_obj[e] = {}
        _dict_obj = _dict_obj[e]
    _dict_obj[list_obj[-1]] = value

    return dict_obj


def get_dict_by_path(dict_obj, path_to_node):
    """
    >>> dict_obj = {"a": {"b": {"c": 1}}}
    >>> get_dict_by_path(dict_obj, ["a", "b", "c"])
    1
    """
    _dict_obj = dict_obj
    for p in path_to_node:
        _dict_obj = _dict_obj[p]
    return _dict_obj


def dict2pv(dict_obj: dict, path_to_node: list = None):
    """
    >>> dict_obj = {"a": {"b": [1, 2], "c": "d"}, "e": 1}
    >>> path, value = dict2pv(dict_obj)
    >>> path
    [['a', 'b'], ['a', 'c'], ['e']]
    >>> value
    [[1, 2], 'd', 1]
    """
    paths = []
    values = []
    path_to_node = [] if path_to_node is None else path_to_node
    for key, value in dict_obj.items():
        if isinstance(value, dict):
            _paths, _values = dict2pv(dict_obj[key], path_to_node + [key])
            paths.extend(_paths)
            values.extend(_values)
        else:
            paths.append(path_to_node + [key])
            values.append(value)
    return paths, values


def get_all_subclass(cls):  # pragma: no cover
    subclass = set()

    def _get_all_subclass(cls, res_set):
        res_set.add(cls)
        for sub_cls in cls.__subclasses__():
            _get_all_subclass(sub_cls, res_set)

    _get_all_subclass(cls, res_set=subclass)
    return subclass


class RegisterNameExistedError(Exception):  # pragma: no cover
    pass


class Register(object):  # pragma: no cover
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
