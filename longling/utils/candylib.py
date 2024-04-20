__all__ = ['as_list', 'dict2pv', 'list2dict', 'get_dict_by_path', 'format_byte_sizeof', 'group_by_n', 'as_ordered_dict']

from collections import OrderedDict


def group_by_n(obj: list, n: int) -> list:
    """
    Examples
    --------
    >>> list_obj = [1, 2, 3, 4, 5, 6]
    >>> group_by_n(list_obj, 3)
    [[1, 2, 3], [4, 5, 6]]
    """
    return [obj[i: i + n] for i in range(0, len(obj), n)]


def as_list(obj) -> list:
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
    >>> as_list((1, 2))
    [1, 2]
    """
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, tuple):
        return list(obj)
    else:
        return [obj]


def as_ordered_dict(dict_data: (dict, OrderedDict), index: (list, None) = None):
    """
    Examples
    --------
    >>> as_ordered_dict({0: 0, 2: 123, 1: 1})
    OrderedDict([(0, 0), (2, 123), (1, 1)])
    >>> as_ordered_dict({0: 0, 2: 123, 1: 1}, [2, 0, 1])
    OrderedDict([(2, 123), (0, 0), (1, 1)])
    >>> as_ordered_dict(OrderedDict([(2, 123), (0, 0), (1, 1)]))
    OrderedDict([(2, 123), (0, 0), (1, 1)])
    """

    if index is None and isinstance(dict_data, OrderedDict):
        return dict_data
    elif index is None and isinstance(dict_data, dict):
        return OrderedDict(dict_data)

    ret = OrderedDict()

    for key in index:
        ret[key] = dict_data[key]

    return ret


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


def format_sizeof(num, suffix='', divisor=1000):
    """
    Code from tqdm

    Formats a number (greater than unity) with SI Order of Magnitude
    prefixes.

    Parameters
    ----------
    num  : float
        Number ( >= 1) to format.
    suffix  : str, optional
        Post-postfix [default: ''].
    divisor  : float, optional
        Divisor between prefixes [default: 1000].

    Returns
    -------
    out  : str
        Number with Order of Magnitude SI unit postfix.

    Examples
    --------
    >>> format_sizeof(800000)
    '800K'
    >>> format_sizeof(40000000000000)
    '40.0T'
    >>> format_sizeof(1000000000000000000000000000)
    '1000.0Y'
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 999.5:
            if abs(num) < 99.95:
                if abs(num) < 9.995:
                    return '{0:1.2f}'.format(num) + unit + suffix
                return '{0:2.1f}'.format(num) + unit + suffix
            return '{0:3.0f}'.format(num) + unit + suffix
        num /= divisor
    return '{0:3.1f}Y'.format(num) + suffix


def format_byte_sizeof(num, suffix='B'):
    """
    Examples
    --------
    >>> format_byte_sizeof(1024)
    '1.00KB'
    """
    return format_sizeof(num, suffix, divisor=1024)


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
